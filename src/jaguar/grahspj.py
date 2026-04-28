from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .config import ImageBandData
from .io import counts_per_mjy_from_ab_zeropoint, counts_to_mjy


def ensure_counts_per_mjy(band: ImageBandData) -> ImageBandData:
    """Return a band with counts_per_mjy populated from AB zeropoint metadata."""

    if band.counts_per_mjy is not None:
        return band
    if band.zeropoint is None:
        raise ValueError(f"Image band {band.filter_name!r} needs counts_per_mjy or zeropoint for grahspj coupling.")
    return replace(band, counts_per_mjy=counts_per_mjy_from_ab_zeropoint(band.zeropoint))


def image_band_photometry_mjy(band: ImageBandData) -> tuple[float, float]:
    """Return summed image photometry and propagated error in mJy."""

    band = ensure_counts_per_mjy(band)
    mask = np.ones_like(band.image, dtype=bool) if band.mask is None else np.asarray(band.mask, dtype=bool)
    flux_counts = float(np.sum(np.where(mask, band.image, 0.0)))
    err_counts = float(np.sqrt(np.sum(np.where(mask, band.noise, 0.0) ** 2)))
    return counts_to_mjy(flux_counts, band.counts_per_mjy), counts_to_mjy(err_counts, band.counts_per_mjy)


def build_grahspj_config_from_image_bands(
    image_bands: Sequence[ImageBandData],
    *,
    object_id: str = "jaguar_source",
    redshift: float = 0.1,
    fit_redshift: bool = False,
    dsps_ssp_fn: str = "../jaxqsofit/tempdata.h5",
    extra_prior_config: dict[str, Any] | None = None,
):
    """Build a grahspj FitConfig from image-summed photometry."""

    from grahspj.config import (
        AGNConfig,
        FitConfig,
        GalaxyConfig,
        InferenceConfig,
        LikelihoodConfig,
        Observation,
        PhotometryData,
    )

    bands = [ensure_counts_per_mjy(band) for band in image_bands]
    fluxes: list[float] = []
    errors: list[float] = []
    for band in bands:
        flux, error = image_band_photometry_mjy(band)
        fluxes.append(flux)
        errors.append(max(error, max(abs(flux) * 0.03, 1.0e-8)))
    return FitConfig(
        observation=Observation(object_id=object_id, redshift=redshift, fit_redshift=fit_redshift),
        photometry=PhotometryData(
            filter_names=[band.filter_name for band in bands],
            fluxes=fluxes,
            errors=errors,
            psf_fwhm_arcsec=[None for _ in bands],
            aperture_diameter_arcsec=[float(max(band.image.shape) * band.pixel_scale) for band in bands],
            photometry_method=["image_sum" for _ in bands],
        ),
        galaxy=GalaxyConfig(dsps_ssp_fn=dsps_ssp_fn, n_wave=512),
        agn=AGNConfig(fit_balmer_continuum=False, fit_feii_broadening=False),
        likelihood=LikelihoodConfig(
            fit_intrinsic_scatter=False,
            intrinsic_scatter_default=1.0e-4,
            variability_uncertainty=False,
            use_absolute_flux_scale_prior=False,
        ),
        inference=InferenceConfig(map_steps=300, num_warmup=100, num_samples=100),
        prior_config=dict(extra_prior_config or {}),
    )
