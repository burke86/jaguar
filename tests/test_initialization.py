from __future__ import annotations

import numpy as np

from jaguar import (
    ImageBandData,
    SceneComponentConfig,
    SedComponentConfig,
    build_grahspj_config_from_image_bands,
    estimate_scene_fluxes_from_pixels,
    initialize_sed_component_amplitudes_from_pixels,
)
from jaguar.initialization import _unit_scene_image
from jaguar.render import psf_unit_flux


def test_pixel_initializer_estimates_point_source_flux_from_local_pixel():
    shape = (21, 21)
    psf = np.ones((5, 5), dtype=float)
    unit = np.asarray(psf_unit_flux(psf, shape, 0.0, 0.0))
    image = unit * 12.0
    band = ImageBandData(image, np.ones(shape), psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=2.0)
    scene = SceneComponentConfig(name="star_image", sed_component="star", kind="point", fit_position=False)

    estimates = estimate_scene_fluxes_from_pixels([band], [scene])

    assert np.isclose(estimates["hsc_i"]["star_image"], 12.0)


def test_pixel_initializer_updates_star_reference_flux_mjy():
    shape = (21, 21)
    psf = np.ones((5, 5), dtype=float)
    unit = np.asarray(psf_unit_flux(psf, shape, 0.0, 0.0))
    image = unit * 8.0
    band = ImageBandData(image, np.ones(shape), psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=4.0)
    sed = [SedComponentConfig(name="star", kind="star", reference_flux_mjy=1.0)]
    scene = [SceneComponentConfig(name="star_image", sed_component="star", kind="point", fit_position=False)]

    initialized = initialize_sed_component_amplitudes_from_pixels([band], sed, scene)

    assert np.isclose(initialized[0].reference_flux_mjy, 2.0)


def test_pixel_initializer_updates_grahspj_component_photometry():
    shape = (21, 21)
    psf = np.ones((5, 5), dtype=float)
    unit = np.asarray(psf_unit_flux(psf, shape, 0.0, 0.0))
    image = unit * 20.0
    band = ImageBandData(image, np.ones(shape), psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=10.0)
    cfg = build_grahspj_config_from_image_bands([band], dsps_ssp_fn="/tmp/not-used.h5")
    sed = [SedComponentConfig(name="agn", kind="agn", grahspj_config=cfg)]
    scene = [SceneComponentConfig(name="agn_image", sed_component="agn", kind="point", fit_position=False)]

    initialized = initialize_sed_component_amplitudes_from_pixels(
        [band],
        sed,
        scene,
        total_grahspj_config=cfg,
    )

    assert np.isclose(initialized[0].grahspj_config.photometry.fluxes[0], 2.0)
    assert initialized[0].grahspj_config.photometry.errors[0] > 0.0


def test_pixel_initializer_sets_host_to_fraction_of_agn_by_default():
    shape = (31, 31)
    psf = np.ones((5, 5), dtype=float)
    unit = np.asarray(psf_unit_flux(psf, shape, 0.0, 0.0))
    image = unit * 20.0
    band = ImageBandData(image, np.ones(shape), psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=10.0)
    cfg = build_grahspj_config_from_image_bands([band], dsps_ssp_fn="/tmp/not-used.h5")
    sed = [
        SedComponentConfig(name="agn", kind="agn", grahspj_config=cfg),
        SedComponentConfig(name="host", kind="host", grahspj_config=cfg),
    ]
    scene = [
        SceneComponentConfig(name="agn_image", sed_component="agn", kind="point", fit_position=False),
        SceneComponentConfig(name="host_image", sed_component="host", kind="sersic", fit_position=False),
    ]

    initialized = initialize_sed_component_amplitudes_from_pixels(
        [band],
        sed,
        scene,
        total_grahspj_config=cfg,
    )

    assert np.isclose(initialized[0].grahspj_config.photometry.fluxes[0], 2.0)
    assert np.isclose(initialized[1].grahspj_config.photometry.fluxes[0], 0.2)
    assert initialized[1].grahspj_config.prior_config == cfg.prior_config


def test_pixel_initializer_can_use_pixel_host_estimate_when_ratio_disabled():
    shape = (31, 31)
    psf = np.ones((5, 5), dtype=float)
    host_scene = SceneComponentConfig(name="host_image", sed_component="host", kind="sersic", fit_position=False)
    host_unit = np.asarray(
        _unit_scene_image(
            host_scene,
            ImageBandData(np.zeros(shape), np.ones(shape), psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=10.0),
        )
    )
    image = host_unit * 50.0
    band = ImageBandData(image, np.ones(shape), psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=10.0)
    cfg = build_grahspj_config_from_image_bands([band], dsps_ssp_fn="/tmp/not-used.h5")
    sed = [SedComponentConfig(name="host", kind="host", grahspj_config=cfg)]

    initialized = initialize_sed_component_amplitudes_from_pixels(
        [band],
        sed,
        [host_scene],
        total_grahspj_config=cfg,
        background_by_filter={"hsc_i": 0.0},
        host_to_agn_initial_flux_ratio=None,
    )

    assert np.isclose(initialized[0].grahspj_config.photometry.fluxes[0], 5.0)
