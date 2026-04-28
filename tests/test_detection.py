from __future__ import annotations

import numpy as np

from jaguar import ImageBandData
from jaguar.detection import (
    DetectedSource,
    SourceDetectionConfig,
    build_components_from_detections,
    build_detection_image,
    detect_sources,
)
from jaguar.grahspj import build_grahspj_config_from_image_bands


def _gaussian(shape, x0, y0, sigma, amp):
    yy, xx = np.indices(shape, dtype=float)
    return amp * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sigma**2)


def _band(image, noise=1.0, psf_sigma=1.0, name="hsc_i", mask=None):
    psf = _gaussian((11, 11), 5, 5, psf_sigma, 1.0)
    return ImageBandData(
        image=np.asarray(image, dtype=float),
        noise=np.ones_like(image, dtype=float) * float(noise),
        psf=psf,
        filter_name=name,
        pixel_scale=0.168,
        counts_per_mjy=1.0,
        mask=mask,
    )


def test_detection_image_uses_inverse_variance_weighting():
    shape = (9, 9)
    image1 = np.ones(shape) * 10.0
    image2 = np.ones(shape) * 100.0
    combined = build_detection_image(
        [
            _band(image1, noise=1.0, name="g"),
            _band(image2, noise=100.0, name="r"),
        ]
    )
    expected = (10.0 / 1.0**2 + 100.0 / 100.0**2) / np.sqrt(1.0 / 1.0**2 + 1.0 / 100.0**2)
    assert np.allclose(combined, expected)


def test_detect_sources_excludes_central_target_and_detects_off_center_source():
    shape = (41, 41)
    image = _gaussian(shape, 20, 20, 1.0, 100.0) + _gaussian(shape, 30, 12, 1.0, 40.0)
    detections = detect_sources(
        [_band(image, noise=1.0)],
        SourceDetectionConfig(threshold=5.0, npixels=3, target_exclusion_radius_pix=5.0),
    )
    assert len(detections) == 1
    assert abs(detections[0].x_pix - 30) < 0.5
    assert abs(detections[0].y_pix - 12) < 0.5


def test_detect_sources_classifies_point_and_extended_sources():
    shape = (61, 61)
    image = _gaussian(shape, 18, 30, 1.0, 80.0) + _gaussian(shape, 45, 30, 4.0, 40.0)
    detections = detect_sources(
        [_band(image, noise=1.0, psf_sigma=1.0)],
        SourceDetectionConfig(
            threshold=5.0,
            npixels=5,
            target_exclusion_radius_pix=0.0,
            extendedness_threshold=1.4,
            classify_stars=True,
        ),
    )
    classes = {source.classification for source in detections}
    assert classes == {"star", "galaxy"}


def test_detect_sources_defaults_point_sources_to_galaxy():
    shape = (41, 41)
    image = _gaussian(shape, 24, 20, 1.0, 80.0)
    detections = detect_sources(
        [_band(image, noise=1.0, psf_sigma=1.0)],
        SourceDetectionConfig(threshold=5.0, npixels=5, target_exclusion_radius_pix=0.0),
    )

    assert len(detections) == 1
    assert detections[0].classification == "galaxy"


def test_detect_sources_deblends_nearby_peak_connected_to_central_source():
    shape = (61, 61)
    image = _gaussian(shape, 30, 30, 5.0, 100.0) + _gaussian(shape, 30, 21, 1.2, 55.0)
    detections = detect_sources(
        [_band(image, noise=1.0, psf_sigma=1.0)],
        SourceDetectionConfig(
            threshold=5.0,
            npixels=5,
            target_exclusion_radius_pix=6.0,
            deblend=True,
            deblend_contrast=0.001,
        ),
    )

    assert any(abs(source.x_pix - 30.0) < 1.0 and abs(source.y_pix - 21.0) < 1.0 for source in detections)


def test_build_components_from_detections_creates_valid_star_and_galaxy_components():
    shape = (11, 11)
    band = _band(np.ones(shape), noise=1.0)
    base_cfg = build_grahspj_config_from_image_bands([band], dsps_ssp_fn="/tmp/not-used.h5")
    detections = [
        DetectedSource(1, 7.0, 5.0, 2.0, 0.0, 10.0, 8.0, 9, 1.0, 1.0, 1.0, "star"),
        DetectedSource(2, 2.0, 5.0, -3.0, 0.0, 20.0, 12.0, 25, 3.0, 1.0, 3.0, "galaxy"),
    ]
    sed, scene = build_components_from_detections(detections, base_cfg, name_prefix="extra")

    assert [component.name for component in sed] == ["extra_1", "extra_2"]
    assert sed[0].kind == "star"
    assert sed[1].kind == "galaxy"
    assert sed[1].grahspj_config.galaxy.fit_host
    assert not sed[1].grahspj_config.agn.fit_agn
    assert scene[0].kind == "point"
    assert scene[1].kind == "sersic"
    assert {component.sed_component for component in scene} == {"extra_1", "extra_2"}
