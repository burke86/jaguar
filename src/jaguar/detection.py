from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from photutils.segmentation import deblend_sources as photutils_deblend_sources
from photutils.segmentation import detect_sources as photutils_detect_sources

from .config import ImageBandData, SceneComponentConfig, SedComponentConfig


@dataclass(frozen=True)
class SourceDetectionConfig:
    """Configuration for quick multi-band scene source detection."""

    threshold: float = 5.0
    npixels: int = 5
    target_exclusion_radius_pix: float = 6.0
    max_sources: int = 8
    extendedness_threshold: float = 1.4
    combination_mode: str = "ivar_snr"
    deblend: bool = True
    deblend_nlevels: int = 32
    deblend_contrast: float = 0.001
    classify_stars: bool = False
    center_sigma_pix: float = 1.0
    default_star_temperature_k: float = 5800.0
    default_reference_flux_mjy: float = 1.0
    default_sersic_reff_arcsec: float = 0.5
    default_sersic_n: float = 2.0


@dataclass(frozen=True)
class DetectedSource:
    """One source detected in the combined JAGUAR detection image."""

    id: int
    x_pix: float
    y_pix: float
    x_offset_pix: float
    y_offset_pix: float
    flux: float
    snr: float
    area: int
    size_pix: float
    psf_size_pix: float
    extendedness: float
    classification: str


def _validate_bands(image_bands: Sequence[ImageBandData]) -> tuple[int, int]:
    if not image_bands:
        raise ValueError("At least one image band is required for source detection.")
    shape = tuple(image_bands[0].image.shape)
    for band in image_bands:
        band.validate()
        if tuple(band.image.shape) != shape:
            raise ValueError("All image bands must have the same image shape for combined detection.")
        if tuple(band.noise.shape) != shape:
            raise ValueError("All image noise maps must match the image shape.")
    return shape


def build_detection_image(
    image_bands: Sequence[ImageBandData],
    config: SourceDetectionConfig | None = None,
) -> np.ndarray:
    """Build a combined detection image from aligned image bands."""

    cfg = SourceDetectionConfig() if config is None else config
    shape = _validate_bands(image_bands)
    numerator = np.zeros(shape, dtype=float)
    weight_sum = np.zeros(shape, dtype=float)
    if cfg.combination_mode not in {"ivar_snr", "mean_snr"}:
        raise ValueError("combination_mode must be 'ivar_snr' or 'mean_snr'.")
    if cfg.combination_mode == "ivar_snr":
        for band in image_bands:
            image = np.asarray(band.image, dtype=float)
            noise = np.asarray(band.noise, dtype=float)
            valid = np.isfinite(image) & np.isfinite(noise) & (noise > 0.0)
            if band.mask is not None:
                valid &= np.asarray(band.mask, dtype=bool)
            weight = np.where(valid, 1.0 / np.square(noise), 0.0)
            numerator += np.where(valid, image, 0.0) * weight
            weight_sum += weight
        return np.where(weight_sum > 0.0, numerator / np.sqrt(weight_sum), 0.0)

    snr_stack = []
    for band in image_bands:
        image = np.asarray(band.image, dtype=float)
        noise = np.asarray(band.noise, dtype=float)
        valid = np.isfinite(image) & np.isfinite(noise) & (noise > 0.0)
        if band.mask is not None:
            valid &= np.asarray(band.mask, dtype=bool)
        snr_stack.append(np.where(valid, image / noise, np.nan))
    return np.nan_to_num(np.nanmean(np.asarray(snr_stack, dtype=float), axis=0), nan=0.0)


def _central_exclusion_mask(shape: tuple[int, int], radius_pix: float) -> np.ndarray:
    if radius_pix <= 0.0:
        return np.zeros(shape, dtype=bool)
    yy, xx = np.indices(shape, dtype=float)
    cx = (shape[1] - 1) / 2.0
    cy = (shape[0] - 1) / 2.0
    return np.hypot(xx - cx, yy - cy) <= float(radius_pix)


def _moment_size(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    yy, xx = np.indices(image.shape, dtype=float)
    weights = np.where(mask, np.clip(image, 0.0, None), 0.0)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        weights = mask.astype(float)
        total = float(np.sum(weights))
    x = float(np.sum(weights * xx) / max(total, 1.0e-30))
    y = float(np.sum(weights * yy) / max(total, 1.0e-30))
    var_x = float(np.sum(weights * np.square(xx - x)) / max(total, 1.0e-30))
    var_y = float(np.sum(weights * np.square(yy - y)) / max(total, 1.0e-30))
    size = float(np.sqrt(max(0.5 * (var_x + var_y), 0.0)))
    return x, y, size, total


def _psf_size_pix(psf: np.ndarray) -> float:
    psf = np.asarray(psf, dtype=float)
    mask = np.isfinite(psf) & (psf > 0.0)
    _x, _y, size, _total = _moment_size(psf, mask)
    return max(size, 1.0e-6)


def detect_sources(
    image_bands: Sequence[ImageBandData],
    config: SourceDetectionConfig | None = None,
) -> list[DetectedSource]:
    """Detect off-center scene sources from a multi-band combined image."""

    cfg = SourceDetectionConfig() if config is None else config
    combined = build_detection_image(image_bands, cfg)
    segm = photutils_detect_sources(
        combined,
        threshold=float(cfg.threshold),
        n_pixels=int(cfg.npixels),
    )
    if segm is None:
        return []
    if cfg.deblend:
        segm = photutils_deblend_sources(
            combined,
            segm,
            n_pixels=int(cfg.npixels),
            n_levels=int(cfg.deblend_nlevels),
            contrast=float(cfg.deblend_contrast),
            progress_bar=False,
        )
    segmentation_data = np.asarray(segm.data, dtype=int)
    labels = [int(label) for label in segm.labels]
    psf_size = float(np.nanmedian([_psf_size_pix(band.psf) for band in image_bands]))
    cx = (combined.shape[1] - 1) / 2.0
    cy = (combined.shape[0] - 1) / 2.0
    detected: list[DetectedSource] = []
    for label in labels:
        mask = np.asarray(segmentation_data == label, dtype=bool)
        x, y, size, positive_flux = _moment_size(combined, mask)
        if np.hypot(x - cx, y - cy) <= float(cfg.target_exclusion_radius_pix):
            continue
        area = int(np.sum(mask))
        flux = float(np.sum(np.where(mask, combined, 0.0)))
        snr = float(positive_flux / np.sqrt(max(area, 1)))
        extendedness = float(size / max(psf_size, 1.0e-6))
        classification = "star" if cfg.classify_stars and extendedness <= float(cfg.extendedness_threshold) else "galaxy"
        detected.append(
            DetectedSource(
                id=int(label),
                x_pix=x,
                y_pix=y,
                x_offset_pix=x - cx,
                y_offset_pix=y - cy,
                flux=flux,
                snr=snr,
                area=area,
                size_pix=size,
                psf_size_pix=psf_size,
                extendedness=extendedness,
                classification=classification,
            )
        )
    detected.sort(key=lambda source: source.snr, reverse=True)
    return detected[: max(int(cfg.max_sources), 0)]


def _galaxy_only_config(base_grahspj_config: Any):
    return replace(
        base_grahspj_config,
        galaxy=replace(base_grahspj_config.galaxy, fit_host=True),
        agn=replace(base_grahspj_config.agn, fit_agn=False),
    )


def build_components_from_detections(
    detections: Sequence[DetectedSource],
    base_grahspj_config: Any,
    config: SourceDetectionConfig | None = None,
    *,
    name_prefix: str = "det",
) -> tuple[list[SedComponentConfig], list[SceneComponentConfig]]:
    """Build SED and scene components for detected extra sources."""

    cfg = SourceDetectionConfig() if config is None else config
    sed_components: list[SedComponentConfig] = []
    scene_components: list[SceneComponentConfig] = []
    for i, source in enumerate(detections, start=1):
        sed_name = f"{name_prefix}_{i}"
        scene_name = f"{sed_name}_image"
        if source.classification == "star":
            sed_components.append(
                SedComponentConfig(
                    name=sed_name,
                    kind="star",
                    temperature_k=cfg.default_star_temperature_k,
                    reference_flux_mjy=cfg.default_reference_flux_mjy,
                )
            )
            scene_components.append(
                SceneComponentConfig(
                    name=scene_name,
                    sed_component=sed_name,
                    kind="point",
                    fixed_center_x_pix=source.x_offset_pix,
                    fixed_center_y_pix=source.y_offset_pix,
                    center_sigma_pix=cfg.center_sigma_pix,
                )
            )
        else:
            sed_components.append(
                SedComponentConfig(
                    name=sed_name,
                    kind="galaxy",
                    spatial="extended",
                    grahspj_config=_galaxy_only_config(base_grahspj_config),
                )
            )
            scene_components.append(
                SceneComponentConfig(
                    name=scene_name,
                    sed_component=sed_name,
                    kind="sersic",
                    fixed_center_x_pix=source.x_offset_pix,
                    fixed_center_y_pix=source.y_offset_pix,
                    center_sigma_pix=cfg.center_sigma_pix,
                    fixed_reff_arcsec=cfg.default_sersic_reff_arcsec,
                    fixed_n_sersic=cfg.default_sersic_n,
                )
            )
    return sed_components, scene_components
