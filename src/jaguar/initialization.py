from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from .config import ImageBandData, ImageFitConfig, SceneComponentConfig, SedComponentConfig
from .grahspj import ensure_counts_per_mjy
from .model import _GALAXY_KINDS
from .render import bounded_psf_padding, convolve_fft_same, pad_psf, psf_unit_flux, sersic_ellipse_unit_flux


def _edge_background(image: np.ndarray) -> float:
    edges = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
    finite = edges[np.isfinite(edges)]
    return float(np.nanmedian(finite)) if finite.size else 0.0


def _scene_sample_pixel(scene: SceneComponentConfig, band: ImageBandData) -> tuple[int, int]:
    """Return a data pixel to use for a rough amplitude estimate."""

    cy = (band.image.shape[0] - 1) / 2.0 + float(scene.fixed_center_y_pix)
    cx = (band.image.shape[1] - 1) / 2.0 + float(scene.fixed_center_x_pix)
    if scene.kind == "sersic":
        offset = max(1.0, min(6.0, float(scene.fixed_reff_arcsec) / float(band.pixel_scale)))
        cx += offset
    x = int(np.clip(np.rint(cx), 0, band.image.shape[1] - 1))
    y = int(np.clip(np.rint(cy), 0, band.image.shape[0] - 1))
    return y, x


def _unit_scene_image(scene: SceneComponentConfig, band: ImageBandData) -> np.ndarray:
    shape = tuple(band.image.shape)
    psf_padding = bounded_psf_padding(tuple(band.psf.shape), shape, band.psf_padding_pixels)
    psf = pad_psf(jnp.asarray(band.psf), psf_padding)
    center_x = float(scene.fixed_center_x_pix)
    center_y = float(scene.fixed_center_y_pix)
    if scene.kind == "point":
        unit = psf_unit_flux(psf, shape, center_x, center_y)
    elif scene.kind == "sersic":
        unit = sersic_ellipse_unit_flux(
            shape,
            float(band.pixel_scale),
            float(scene.fixed_reff_arcsec),
            float(scene.fixed_n_sersic),
            float(scene.fixed_e1),
            float(scene.fixed_e2),
            center_x,
            center_y,
        )
        unit = convolve_fft_same(unit, psf)
    else:  # pragma: no cover - config validation catches this
        raise ValueError(f"Unsupported scene component kind {scene.kind!r}.")
    return np.asarray(unit, dtype=float)


def estimate_scene_fluxes_from_pixels(
    image_bands: Sequence[ImageBandData],
    scene_components: Sequence[SceneComponentConfig],
    *,
    background_by_filter: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate scene-component count fluxes from local pixel values.

    This intentionally avoids fitting. For each component and band, JAGUAR
    renders a unit-flux image, reads one representative data pixel, and divides
    the background-subtracted pixel value by the unit template value there.
    """

    bands = [ensure_counts_per_mjy(band) for band in image_bands]
    estimates: dict[str, dict[str, float]] = {}
    for band in bands:
        background = (
            float(background_by_filter[band.filter_name])
            if background_by_filter is not None and band.filter_name in background_by_filter
            else _edge_background(np.asarray(band.image, dtype=float))
        )
        band_estimates: dict[str, float] = {}
        image = np.asarray(band.image, dtype=float)
        for scene in scene_components:
            unit = _unit_scene_image(scene, band)
            y, x = _scene_sample_pixel(scene, band)
            pixel = float(image[y, x] - background)
            unit_pixel = float(unit[y, x])
            flux = pixel / max(unit_pixel, 1.0e-30)
            band_estimates[scene.name] = float(max(flux, 1.0e-12))
        estimates[band.filter_name] = band_estimates
    return estimates


def estimate_sed_fluxes_from_pixels(
    image_bands: Sequence[ImageBandData],
    sed_components: Sequence[SedComponentConfig],
    scene_components: Sequence[SceneComponentConfig],
    *,
    background_by_filter: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate per-SED-component mJy fluxes from local component pixels."""

    scene_fluxes = estimate_scene_fluxes_from_pixels(
        image_bands,
        scene_components,
        background_by_filter=background_by_filter,
    )
    component_names = {component.name for component in sed_components}
    sed_fluxes = {name: {} for name in component_names}
    for band in image_bands:
        band = ensure_counts_per_mjy(band)
        for scene in scene_components:
            if scene.sed_component not in component_names:
                continue
            counts = scene_fluxes[band.filter_name][scene.name]
            sed_fluxes[scene.sed_component][band.filter_name] = (
                sed_fluxes[scene.sed_component].get(band.filter_name, 0.0)
                + counts / float(band.counts_per_mjy)
            )
    return sed_fluxes


def _with_component_photometry(
    cfg: Any,
    image_bands: Sequence[ImageBandData],
    flux_by_filter: Mapping[str, float],
    *,
    kind: str,
    total_flux_by_filter: Mapping[str, float] | None,
) -> Any:
    if cfg is None:
        return None
    del kind, total_flux_by_filter
    fluxes = [max(float(flux_by_filter.get(band.filter_name, 0.0)), 1.0e-12) for band in image_bands]
    errors = [max(abs(flux) * 0.2, 1.0e-8) for flux in fluxes]
    photometry = replace(cfg.photometry, fluxes=fluxes, errors=errors)
    return replace(cfg, photometry=photometry)


def initialize_sed_component_amplitudes_from_pixels(
    image_bands: Sequence[ImageBandData],
    sed_components: Sequence[SedComponentConfig],
    scene_components: Sequence[SceneComponentConfig],
    *,
    total_grahspj_config: Any | None = None,
    background_by_filter: Mapping[str, float] | None = None,
    host_to_agn_initial_flux_ratio: float | None = 0.1,
) -> list[SedComponentConfig]:
    """Return SED components with pixel-informed rough amplitude settings."""

    bands = [ensure_counts_per_mjy(band) for band in image_bands]
    sed_fluxes = estimate_sed_fluxes_from_pixels(
        bands,
        sed_components,
        scene_components,
        background_by_filter=background_by_filter,
    )
    total_flux_by_filter = None
    if total_grahspj_config is not None:
        total_flux_by_filter = {
            name: float(flux)
            for name, flux in zip(total_grahspj_config.photometry.filter_names, total_grahspj_config.photometry.fluxes, strict=False)
        }
    initialized: list[SedComponentConfig] = []
    filter_names = [band.filter_name for band in bands]
    agn_component_names = {component.name for component in sed_components if component.kind == "agn"}
    agn_flux_by_filter: dict[str, float] = {}
    for agn_name in agn_component_names:
        for name, flux in sed_fluxes.get(agn_name, {}).items():
            agn_flux_by_filter[name] = agn_flux_by_filter.get(name, 0.0) + float(flux)
    use_host_to_agn_ratio = (
        host_to_agn_initial_flux_ratio is not None
        and np.isfinite(float(host_to_agn_initial_flux_ratio))
        and float(host_to_agn_initial_flux_ratio) > 0.0
        and any(flux > 0.0 for flux in agn_flux_by_filter.values())
    )
    for component in sed_components:
        flux_by_filter = sed_fluxes.get(component.name, {})
        if component.kind in _GALAXY_KINDS and use_host_to_agn_ratio:
            flux_by_filter = {
                name: max(float(host_to_agn_initial_flux_ratio) * float(agn_flux_by_filter.get(name, 0.0)), 1.0e-12)
                for name in filter_names
            }
        positive = np.asarray([float(flux_by_filter.get(name, np.nan)) for name in filter_names], dtype=float)
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if component.kind == "star" and positive.size:
            if component.reference_filter_name in filter_names:
                reference_flux = float(flux_by_filter.get(component.reference_filter_name, np.nan))
            else:
                reference_flux = float(np.nanmedian(positive))
            initialized.append(replace(component, reference_flux_mjy=max(reference_flux, 1.0e-12)))
            continue
        if component.kind == "agn" or component.kind in _GALAXY_KINDS:
            initialized.append(
                replace(
                    component,
                    grahspj_config=_with_component_photometry(
                        component.grahspj_config,
                        bands,
                        flux_by_filter,
                        kind=component.kind,
                        total_flux_by_filter=total_flux_by_filter,
                    ),
                )
            )
            continue
        initialized.append(component)
    return initialized
