from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from photutils.segmentation import detect_sources as photutils_detect_sources

from .config import ImageBandData


GALIGHT_HSC_DRIVE_ID = "1ZO9-HzV8K60ijYWK98jGoSoZHjIGW5Lc"
GALIGHT_HSC_QSO_IMAGE = "example_data/HSC/QSO/000017.88+002612.6_HSC-I.fits"
GALIGHT_HSC_QSO_PSF = "example_data/HSC/QSO/000017.88+002612.6_HSC-I_psf.fits"
LEGACY_SURVEY_DR10_BASE_URL = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10"


@dataclass(frozen=True)
class PsfCandidate:
    """Compact source selected as an empirical PSF candidate."""

    x_pix: float
    y_pix: float
    flux: float
    size_pix: float
    peak: float
    fwhm_pix: float = np.nan
    fwhm_fractional_scatter: float = np.nan
    is_gaia_star: bool = False


@dataclass(frozen=True)
class EmpiricalPsfConfig:
    """Selection settings for empirical PSF construction."""

    psf_size: int = 51
    psf_search_radius: int = 600
    threshold_sigma: float = 4.0
    npixels: int = 3
    min_sources: int = 1
    max_sources: int = 8
    candidate_pool_size: int = 32
    target_exclusion_radius_pix: float = 12.0
    min_size_pix: float = 0.2
    max_size_pix: float = 8.0
    max_peak_percentile: float = 100.0
    max_edge_flux_fraction: float | None = 0.10
    saturation_peak_fraction: float = 0.95
    max_saturated_pixels: int | None = 4
    isolation_radius_pix: float | None = None
    max_neighbor_flux_ratio: float = 0.05
    max_fwhm_fractional_scatter: float | None = 0.1
    prefer_common_stars: bool = True
    min_common_bands: int | None = None
    common_match_radius_arcsec: float = 1.0
    require_gaia_match: bool = False
    annotate_gaia_matches: bool = False
    gaia_match_radius_arcsec: float = 1.0
    gaia_xmatch_timeout: float | None = 10.0
    psf_padding_pixels: int = 20
    print_diagnostics: bool = True


@dataclass(frozen=True)
class EmpiricalPsfBandResult:
    """Empirical PSF products for one image band."""

    band_code: str
    filter_name: str
    image_path: Path
    invvar_path: Path
    psf: np.ndarray
    candidates: list[PsfCandidate]
    selected_candidates: list[PsfCandidate]
    search_image: np.ndarray
    search_target_pixel: tuple[float, float]
    search_wcs: WCS
    full_target_pixel: tuple[float, float]
    search_origin: tuple[int, int]
    psf_uncertainty: np.ndarray | None = None


@dataclass(frozen=True)
class EmpiricalPsfResult:
    """Multi-band empirical PSF products."""

    brick: str
    image_bands: list[ImageBandData]
    bands: dict[str, EmpiricalPsfBandResult]
    common_star_groups: list[dict[str, PsfCandidate]]
    config: EmpiricalPsfConfig


def counts_per_mjy_from_ab_zeropoint(zeropoint: float) -> float:
    """Return image counts per mJy for an AB zeropoint."""

    return float(10 ** ((float(zeropoint) - 16.4) / 2.5))


def counts_to_mjy(counts: float, counts_per_mjy: float) -> float:
    """Convert image counts to mJy."""

    return float(counts) / float(counts_per_mjy)


def nanomaggy_counts_per_mjy() -> float:
    """Return the number of nanomaggies in one mJy."""

    return counts_per_mjy_from_ab_zeropoint(22.5)


def _download_url(url: str, path: Path, *, overwrite: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    with urlopen(url) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def legacy_survey_bricks_url(*, base_url: str = LEGACY_SURVEY_DR10_BASE_URL, region: str = "south") -> str:
    """Return the DR10 Legacy Survey brick-summary table URL."""

    return f"{base_url.rstrip('/')}/{region}/survey-bricks-dr10-{region}.fits.gz"


def download_legacy_survey_bricks_table(
    output_dir: str | Path,
    *,
    base_url: str = LEGACY_SURVEY_DR10_BASE_URL,
    region: str = "south",
    overwrite: bool = False,
) -> Path:
    """Download/cache the DR10 Legacy Survey brick-summary table."""

    output_dir = Path(output_dir)
    path = output_dir / f"survey-bricks-dr10-{region}.fits.gz"
    return _download_url(legacy_survey_bricks_url(base_url=base_url, region=region), path, overwrite=overwrite)


def _column(data: Any, name: str) -> np.ndarray:
    names = {col.lower(): col for col in data.names}
    return np.asarray(data[names[name.lower()]])


def find_legacy_survey_brick(ra: float, dec: float, bricks_fits: str | Path) -> str:
    """Return the DR10 brick name covering an RA/Dec coordinate."""

    with fits.open(bricks_fits) as hdul:
        data = hdul[1].data
        brickname = _column(data, "brickname")
        ra_col = _column(data, "ra")
        dec_col = _column(data, "dec")
        ra1 = _column(data, "ra1")
        ra2 = _column(data, "ra2")
        dec1 = _column(data, "dec1")
        dec2 = _column(data, "dec2")
    ra = float(ra) % 360.0
    dec = float(dec)
    normal_ra = ra1 <= ra2
    in_ra = np.where(normal_ra, (ra >= ra1) & (ra <= ra2), (ra >= ra1) | (ra <= ra2))
    in_dec = (dec >= dec1) & (dec <= dec2)
    matches = np.flatnonzero(in_ra & in_dec)
    if matches.size == 0:
        raise ValueError(f"No DR10 Legacy Survey brick contains RA={ra}, Dec={dec}.")
    if matches.size > 1:
        distances = np.square(((ra_col[matches] - ra + 180.0) % 360.0) - 180.0) + np.square(dec_col[matches] - dec)
        index = int(matches[int(np.argmin(distances))])
    else:
        index = int(matches[0])
    value = brickname[index]
    return value.decode("utf-8").strip() if isinstance(value, bytes) else str(value).strip()


def legacy_survey_coadd_url(
    brick: str,
    band: str,
    *,
    kind: str = "image",
    base_url: str = LEGACY_SURVEY_DR10_BASE_URL,
    region: str = "south",
) -> str:
    """Return a static DR10 coadd file URL for a brick and band."""

    brick = str(brick)
    band = str(band)
    prefix = brick[:3]
    return f"{base_url.rstrip('/')}/{region}/coadd/{prefix}/{brick}/legacysurvey-{brick}-{kind}-{band}.fits.fz"


def download_legacy_survey_coadd_band(
    output_dir: str | Path,
    *,
    brick: str,
    band: str,
    base_url: str = LEGACY_SURVEY_DR10_BASE_URL,
    region: str = "south",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Download/cache DR10 coadd image and inverse-variance files."""

    output_dir = Path(output_dir)
    brick = str(brick)
    band = str(band)
    prefix = brick[:3]
    brick_dir = output_dir / "dr10" / region / "coadd" / prefix / brick
    image_path = brick_dir / f"legacysurvey-{brick}-image-{band}.fits.fz"
    invvar_path = brick_dir / f"legacysurvey-{brick}-invvar-{band}.fits.fz"
    _download_url(legacy_survey_coadd_url(brick, band, kind="image", base_url=base_url, region=region), image_path, overwrite=overwrite)
    _download_url(legacy_survey_coadd_url(brick, band, kind="invvar", base_url=base_url, region=region), invvar_path, overwrite=overwrite)
    return image_path, invvar_path


def _mad_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    return max(1.4826 * mad, 1.0e-12)


def _centroid_and_size(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float, float]:
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
    peak = float(np.nanmax(np.where(mask, image, np.nan)))
    return x, y, size, total, peak


def _centroid_and_size_for_label(
    image: np.ndarray,
    segmentation: np.ndarray,
    label: int,
    source_slice,
) -> tuple[float, float, float, float, float]:
    y_slice, x_slice = source_slice
    cutout = image[y_slice, x_slice]
    mask = segmentation[y_slice, x_slice] == int(label)
    x, y, size, total, peak = _centroid_and_size(cutout, mask)
    return x + float(x_slice.start or 0), y + float(y_slice.start or 0), size, total, peak


def _profile_fwhm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return np.nan
    values = values - np.nanmin(values)
    peak = float(np.nanmax(values))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.nan
    above = np.flatnonzero(values >= 0.5 * peak)
    if above.size < 2:
        return np.nan
    return float(above[-1] - above[0] + 1)


def _cutout_fwhm_metrics(stamp: np.ndarray) -> tuple[float, float]:
    stamp = np.asarray(stamp, dtype=float)
    if stamp.ndim != 2 or min(stamp.shape) < 3:
        return np.nan, np.nan
    y0, x0 = np.unravel_index(int(np.nanargmax(stamp)), stamp.shape)
    max_radius = min(x0, y0, stamp.shape[1] - x0 - 1, stamp.shape[0] - y0 - 1)
    if max_radius < 1:
        return np.nan, np.nan
    offsets = np.arange(-max_radius, max_radius + 1)
    profiles = [
        stamp[y0, x0 + offsets],
        stamp[y0 + offsets, x0],
        stamp[y0 + offsets, x0 + offsets],
        stamp[y0 - offsets, x0 + offsets],
    ]
    fwhms = np.asarray([_profile_fwhm(profile) for profile in profiles], dtype=float)
    fwhms = fwhms[np.isfinite(fwhms) & (fwhms > 0.0)]
    if fwhms.size == 0:
        return np.nan, np.nan
    mean = float(np.nanmean(fwhms))
    scatter = float(np.nanstd(fwhms) / max(mean, 1.0e-30))
    return mean, scatter


def _candidate_world_coordinates(candidate: PsfCandidate, wcs: WCS) -> tuple[float, float] | None:
    ra, dec = wcs.pixel_to_world_values(float(candidate.x_pix), float(candidate.y_pix))
    ra = float(np.asarray(ra))
    dec = float(np.asarray(dec))
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    return ra, dec


def _with_gaia_xmatch(
    candidates: list[PsfCandidate],
    *,
    wcs: WCS | None,
    radius_arcsec: float,
    timeout: float | None,
    catalog: str = "vizier:I/355/gaiadr3",
) -> tuple[list[PsfCandidate], bool]:
    if not candidates or wcs is None:
        return candidates, False
    try:
        import astropy.units as u
        from astropy.table import Table
        from astroquery.xmatch import XMatch
    except Exception:
        return candidates, False

    rows = []
    for index, candidate in enumerate(candidates):
        coords = _candidate_world_coordinates(candidate, wcs)
        if coords is None:
            continue
        ra, dec = coords
        rows.append((index, ra, dec))
    if not rows:
        return candidates, False

    query_table = Table(rows=rows, names=("candidate_index", "ra", "dec"))
    old_timeout = getattr(XMatch, "TIMEOUT", None)
    had_timeout = hasattr(XMatch, "TIMEOUT")
    if timeout is not None:
        XMatch.TIMEOUT = float(timeout)
    try:
        matched = XMatch.query(
            cat1=query_table,
            cat2=catalog,
            max_distance=float(radius_arcsec) * u.arcsec,
            colRA1="ra",
            colDec1="dec",
        )
    except Exception:
        return candidates, False
    finally:
        if timeout is not None:
            if had_timeout:
                XMatch.TIMEOUT = old_timeout
            else:
                try:
                    delattr(XMatch, "TIMEOUT")
                except Exception:
                    pass

    matched_indices = set()
    if "candidate_index" in matched.colnames:
        matched_indices = {int(value) for value in np.asarray(matched["candidate_index"])}
    return [
        replace(candidate, is_gaia_star=index in matched_indices)
        for index, candidate in enumerate(candidates)
    ], True


def _select_empirical_psf_candidates(
    source_image: np.ndarray,
    *,
    target_pixel: tuple[float, float] | None,
    wcs: WCS | None,
    psf_size: int,
    threshold_sigma: float,
    npixels: int,
    min_sources: int,
    max_sources: int,
    target_exclusion_radius_pix: float,
    min_size_pix: float,
    max_size_pix: float,
    max_peak_percentile: float,
    max_edge_flux_fraction: float | None,
    saturation_peak_fraction: float,
    max_saturated_pixels: int | None,
    isolation_radius_pix: float | None,
    max_neighbor_flux_ratio: float,
    max_fwhm_fractional_scatter: float | None,
    gaia_match_radius_arcsec: float,
    gaia_xmatch_timeout: float | None,
    require_gaia_match: bool,
    diagnostics: Counter[str] | None = None,
) -> list[PsfCandidate]:
    sigma = _mad_std(source_image)
    threshold = float(threshold_sigma) * sigma
    segm = photutils_detect_sources(source_image, threshold=threshold, n_pixels=int(npixels))
    if segm is None:
        if diagnostics is not None:
            diagnostics["detected_segments"] = 0
            diagnostics["accepted"] = 0
        return []
    labels = [int(label) for label in segm.labels]
    segmentation = np.asarray(segm.data, dtype=int)
    if target_pixel is None:
        target_pixel = ((source_image.shape[1] - 1) / 2.0, (source_image.shape[0] - 1) / 2.0)
    tx, ty = target_pixel
    half = psf_size // 2
    isolation_radius = float(psf_size) / 2.0 if isolation_radius_pix is None else float(isolation_radius_pix)
    all_sources: list[tuple[int, float, float, float, float, float]] = []
    for label, source_slice in zip(labels, segm.slices):
        x, y, size, flux, peak = _centroid_and_size_for_label(source_image, segmentation, label, source_slice)
        all_sources.append((label, x, y, size, flux, peak))
    if diagnostics is not None:
        diagnostics["detected_segments"] = len(all_sources)

    candidates: list[PsfCandidate] = []
    for label, x, y, size, flux, peak in all_sources:
        if np.hypot(x - tx, y - ty) <= float(target_exclusion_radius_pix):
            if diagnostics is not None:
                diagnostics["target_exclusion"] += 1
            continue
        if not min_size_pix <= size <= max_size_pix:
            if diagnostics is not None:
                diagnostics["size"] += 1
            continue
        if x - half < 0 or y - half < 0 or x + half + 1 > source_image.shape[1] or y + half + 1 > source_image.shape[0]:
            if diagnostics is not None:
                diagnostics["stamp_outside"] += 1
            continue
        if flux <= 0.0:
            if diagnostics is not None:
                diagnostics["nonpositive_flux"] += 1
            continue
        stamp = _cutout(source_image, (x, y), half)
        inner_radius = float(half) + 1.0
        outer_radius = max(inner_radius + 2.0, float(half) * 1.8)
        stamp_for_quality = np.clip(stamp - _local_annulus_background(source_image, (x, y), inner_radius, outer_radius), 0.0, None)
        if max_edge_flux_fraction is not None:
            if _edge_flux_fraction(stamp_for_quality) > float(max_edge_flux_fraction):
                if diagnostics is not None:
                    diagnostics["edge_flux"] += 1
                continue
        if max_saturated_pixels is not None:
            if _saturated_core_pixel_count(stamp_for_quality, saturation_peak_fraction) > int(max_saturated_pixels):
                if diagnostics is not None:
                    diagnostics["saturated_core"] += 1
                continue
        fwhm_pix, fwhm_fractional_scatter = _cutout_fwhm_metrics(stamp)
        if max_fwhm_fractional_scatter is not None:
            if not np.isfinite(fwhm_fractional_scatter) or fwhm_fractional_scatter > float(max_fwhm_fractional_scatter):
                if diagnostics is not None:
                    diagnostics["fwhm_scatter"] += 1
                continue
        if isolation_radius > 0.0:
            contaminated = any(
                other_label != label
                and other_flux > max(float(max_neighbor_flux_ratio) * flux, 0.0)
                and np.hypot(other_x - x, other_y - y) <= isolation_radius
                for other_label, other_x, other_y, _other_size, other_flux, _other_peak in all_sources
            )
            if contaminated:
                if diagnostics is not None:
                    diagnostics["neighbor"] += 1
                continue
        candidates.append(
            PsfCandidate(
                x_pix=x,
                y_pix=y,
                flux=flux,
                size_pix=size,
                peak=peak,
                fwhm_pix=fwhm_pix,
                fwhm_fractional_scatter=fwhm_fractional_scatter,
            )
        )
    if candidates:
        peak_limit = float(np.nanpercentile([source.peak for source in candidates], max_peak_percentile))
        filtered = [source for source in candidates if source.peak <= peak_limit]
        if len(filtered) >= int(min_sources):
            if diagnostics is not None:
                diagnostics["peak_percentile"] += len(candidates) - len(filtered)
            candidates = filtered
    candidates.sort(key=lambda source: source.flux, reverse=True)
    candidates, gaia_checked = _with_gaia_xmatch(
        candidates,
        wcs=wcs,
        radius_arcsec=gaia_match_radius_arcsec,
        timeout=gaia_xmatch_timeout,
    )
    if require_gaia_match and gaia_checked:
        before_gaia = len(candidates)
        candidates = [candidate for candidate in candidates if candidate.is_gaia_star]
        if diagnostics is not None:
            diagnostics["gaia_match"] += before_gaia - len(candidates)
    if diagnostics is not None:
        if len(candidates) > int(max_sources):
            diagnostics["candidate_pool_limit"] += len(candidates) - int(max_sources)
        diagnostics["accepted"] = min(len(candidates), int(max_sources))
    return candidates[: int(max_sources)]


def find_empirical_psf_candidates(
    image: np.ndarray,
    *,
    target_pixel: tuple[float, float] | None = None,
    wcs: WCS | None = None,
    psf_size: int = 25,
    threshold_sigma: float = 8.0,
    npixels: int = 5,
    min_sources: int = 1,
    max_sources: int = 8,
    target_exclusion_radius_pix: float = 20.0,
    min_size_pix: float = 0.5,
    max_size_pix: float = 6.0,
    max_peak_percentile: float = 99.5,
    max_edge_flux_fraction: float | None = 0.10,
    saturation_peak_fraction: float = 0.95,
    max_saturated_pixels: int | None = 4,
    isolation_radius_pix: float | None = None,
    max_neighbor_flux_ratio: float = 0.05,
    max_fwhm_fractional_scatter: float | None = 0.1,
    gaia_match_radius_arcsec: float = 1.0,
    gaia_xmatch_timeout: float | None = 10.0,
    require_gaia_match: bool = False,
    diagnostics: Counter[str] | None = None,
) -> list[PsfCandidate]:
    """Return compact sources that would be used to build an empirical PSF."""

    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D.")
    if psf_size % 2 == 0:
        raise ValueError("psf_size must be odd.")
    background = float(np.nanmedian(image))
    source_image = image - background
    return _select_empirical_psf_candidates(
        source_image,
        target_pixel=target_pixel,
        wcs=wcs,
        psf_size=psf_size,
        threshold_sigma=threshold_sigma,
        npixels=npixels,
        min_sources=min_sources,
        max_sources=max_sources,
        target_exclusion_radius_pix=target_exclusion_radius_pix,
        min_size_pix=min_size_pix,
        max_size_pix=max_size_pix,
        max_peak_percentile=max_peak_percentile,
        max_edge_flux_fraction=max_edge_flux_fraction,
        saturation_peak_fraction=saturation_peak_fraction,
        max_saturated_pixels=max_saturated_pixels,
        isolation_radius_pix=isolation_radius_pix,
        max_neighbor_flux_ratio=max_neighbor_flux_ratio,
        max_fwhm_fractional_scatter=max_fwhm_fractional_scatter,
        gaia_match_radius_arcsec=gaia_match_radius_arcsec,
        gaia_xmatch_timeout=gaia_xmatch_timeout,
        require_gaia_match=require_gaia_match,
        diagnostics=diagnostics,
    )


def construct_empirical_psf(
    image: np.ndarray,
    *,
    target_pixel: tuple[float, float] | None = None,
    wcs: WCS | None = None,
    psf_size: int = 25,
    threshold_sigma: float = 8.0,
    npixels: int = 5,
    min_sources: int = 1,
    max_sources: int = 8,
    target_exclusion_radius_pix: float = 20.0,
    min_size_pix: float = 0.5,
    max_size_pix: float = 6.0,
    max_peak_percentile: float = 99.5,
    max_edge_flux_fraction: float | None = 0.10,
    saturation_peak_fraction: float = 0.95,
    max_saturated_pixels: int | None = 4,
    isolation_radius_pix: float | None = None,
    max_neighbor_flux_ratio: float = 0.05,
    max_fwhm_fractional_scatter: float | None = 0.1,
    gaia_match_radius_arcsec: float = 1.0,
    gaia_xmatch_timeout: float | None = 10.0,
    require_gaia_match: bool = False,
    weight_image: np.ndarray | None = None,
    return_uncertainty: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build a normalized empirical PSF from bright compact cutout sources."""

    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D.")
    if weight_image is not None:
        weight_image = np.asarray(weight_image, dtype=float)
        if weight_image.shape != image.shape:
            raise ValueError("weight_image must match image shape.")
    if psf_size % 2 == 0:
        raise ValueError("psf_size must be odd.")
    background = float(np.nanmedian(image))
    source_image = image - background
    candidates = _select_empirical_psf_candidates(
        source_image,
        target_pixel=target_pixel,
        wcs=wcs if require_gaia_match else None,
        psf_size=psf_size,
        threshold_sigma=threshold_sigma,
        npixels=npixels,
        min_sources=min_sources,
        max_sources=max_sources,
        target_exclusion_radius_pix=target_exclusion_radius_pix,
        min_size_pix=min_size_pix,
        max_size_pix=max_size_pix,
        max_peak_percentile=max_peak_percentile,
        max_edge_flux_fraction=max_edge_flux_fraction,
        saturation_peak_fraction=saturation_peak_fraction,
        max_saturated_pixels=max_saturated_pixels,
        isolation_radius_pix=isolation_radius_pix,
        max_neighbor_flux_ratio=max_neighbor_flux_ratio,
        max_fwhm_fractional_scatter=max_fwhm_fractional_scatter,
        gaia_match_radius_arcsec=gaia_match_radius_arcsec,
        gaia_xmatch_timeout=gaia_xmatch_timeout,
        require_gaia_match=require_gaia_match,
    )
    if not candidates:
        raise ValueError("No compact sources were detected for empirical PSF construction.")
    stamps: list[np.ndarray] = []
    weights: list[np.ndarray | None] = []
    half = psf_size // 2
    for candidate in candidates:
        normalized = _normalize_psf_stamp(
            source_image,
            candidate,
            half,
            weight_image=weight_image,
            max_edge_flux_fraction=max_edge_flux_fraction,
            saturation_peak_fraction=saturation_peak_fraction,
            max_saturated_pixels=max_saturated_pixels,
        )
        if normalized is not None:
            stamp, weight = normalized
            stamps.append(stamp)
            weights.append(weight)
    if len(stamps) < int(min_sources):
        raise ValueError(f"Only found {len(stamps)} usable PSF source(s); require {min_sources}.")
    psf, psf_uncertainty = _stack_normalized_psf_stamps(stamps, weights)
    psf = np.clip(psf, 0.0, None)
    total = float(np.sum(psf))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Empirical PSF stack has non-positive total flux.")
    psf = psf / total
    psf_uncertainty = np.asarray(psf_uncertainty, dtype=float) / total
    if return_uncertainty:
        return psf, psf_uncertainty
    return psf


def _local_annulus_background(image: np.ndarray, center_xy: tuple[float, float], inner_radius: float, outer_radius: float) -> float:
    """Return a local median background from an annulus around one source."""

    cx, cy = center_xy
    margin = int(np.ceil(outer_radius))
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    x0 = max(0, cx_i - margin)
    x1 = min(image.shape[1], cx_i + margin + 1)
    y0 = max(0, cy_i - margin)
    y1 = min(image.shape[0], cy_i + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
    radius = np.hypot(xx + x0 - float(cx), yy + y0 - float(cy))
    annulus = (radius > float(inner_radius)) & (radius <= float(outer_radius))
    values = np.asarray(image[y0:y1, x0:x1], dtype=float)[annulus]
    values = values[np.isfinite(values)]
    return float(np.nanmedian(values)) if values.size else 0.0


def _edge_flux_fraction(stamp: np.ndarray) -> float:
    """Return the fraction of positive stamp flux on the outer pixel border."""

    stamp = np.asarray(stamp, dtype=float)
    positive = np.where(np.isfinite(stamp) & (stamp > 0.0), stamp, 0.0)
    total = float(np.sum(positive))
    if not np.isfinite(total) or total <= 0.0:
        return np.inf
    edge = np.concatenate([positive[0], positive[-1], positive[1:-1, 0], positive[1:-1, -1]])
    return float(np.sum(edge) / total)


def _saturated_core_pixel_count(stamp: np.ndarray, peak_fraction: float) -> int:
    """Return the number of pixels close enough to the peak to indicate clipping."""

    stamp = np.asarray(stamp, dtype=float)
    finite = stamp[np.isfinite(stamp)]
    if finite.size == 0:
        return 0
    peak = float(np.nanmax(finite))
    if not np.isfinite(peak) or peak <= 0.0:
        return 0
    return int(np.count_nonzero(stamp >= float(peak_fraction) * peak))


def _normalize_psf_stamp(
    source_image: np.ndarray,
    candidate: PsfCandidate,
    half_size: int,
    *,
    weight_image: np.ndarray | None,
    max_edge_flux_fraction: float | None,
    saturation_peak_fraction: float,
    max_saturated_pixels: int | None,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    inner_radius = float(half_size) + 1.0
    outer_radius = max(inner_radius + 2.0, float(half_size) * 1.8)
    local_background = _local_annulus_background(source_image, (candidate.x_pix, candidate.y_pix), inner_radius, outer_radius)
    stamp = _cutout(source_image, (candidate.x_pix, candidate.y_pix), half_size) - local_background
    stamp = np.clip(stamp, 0.0, None)
    if max_edge_flux_fraction is not None and _edge_flux_fraction(stamp) > float(max_edge_flux_fraction):
        return None
    if max_saturated_pixels is not None and _saturated_core_pixel_count(stamp, saturation_peak_fraction) > int(max_saturated_pixels):
        return None
    total = float(np.sum(stamp))
    if not np.isfinite(total) or total <= 0.0:
        return None
    normalized = stamp / total
    if weight_image is None:
        return normalized, None
    weights = _cutout(weight_image, (candidate.x_pix, candidate.y_pix), half_size)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    return normalized, weights * total * total


def _stack_normalized_psf_stamps(stamps: list[np.ndarray], weights: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    stamp_array = np.asarray(stamps, dtype=float)
    if not weights or all(weight is None for weight in weights):
        psf = np.nanmedian(stamp_array, axis=0)
        if len(stamp_array) <= 1:
            return psf, np.zeros_like(psf)
        return psf, 1.253 * np.nanstd(stamp_array, axis=0) / np.sqrt(float(len(stamp_array)))
    weight_array = np.asarray(
        [np.ones_like(stamps[index], dtype=float) if weight is None else np.asarray(weight, dtype=float) for index, weight in enumerate(weights)],
        dtype=float,
    )
    numerator = np.sum(stamp_array * weight_array, axis=0)
    denominator = np.sum(weight_array, axis=0)
    fallback = np.nanmedian(stamp_array, axis=0)
    psf = np.where(denominator > 0.0, numerator / denominator, fallback)
    formal = np.where(denominator > 0.0, 1.0 / np.sqrt(denominator), 0.0)
    weighted_scatter = np.sqrt(np.where(denominator > 0.0, np.sum(weight_array * (stamp_array - psf) ** 2, axis=0) / denominator, 0.0))
    weight_sq_sum = np.sum(weight_array * weight_array, axis=0)
    n_eff = np.where(weight_sq_sum > 0.0, denominator * denominator / weight_sq_sum, 1.0)
    scatter_uncertainty = weighted_scatter / np.sqrt(np.maximum(n_eff, 1.0))
    fallback_uncertainty = np.zeros_like(fallback) if len(stamp_array) <= 1 else np.nanstd(stamp_array, axis=0) / np.sqrt(float(len(stamp_array)))
    uncertainty = np.where(denominator > 0.0, np.sqrt(formal * formal + scatter_uncertainty * scatter_uncertainty), fallback_uncertainty)
    return psf, uncertainty


def _construct_empirical_psf_from_candidates(
    source_image: np.ndarray,
    candidates: list[PsfCandidate],
    *,
    target_pixel: tuple[float, float] | None,
    psf_size: int,
    min_sources: int,
    weight_image: np.ndarray | None = None,
    max_edge_flux_fraction: float | None = 0.10,
    saturation_peak_fraction: float = 0.95,
    max_saturated_pixels: int | None = 4,
    return_uncertainty: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if not candidates:
        raise ValueError("No compact sources were selected for empirical PSF construction.")
    half = int(psf_size) // 2
    stamps: list[np.ndarray] = []
    weights: list[np.ndarray | None] = []
    for candidate in candidates:
        normalized = _normalize_psf_stamp(
            source_image,
            candidate,
            half,
            weight_image=weight_image,
            max_edge_flux_fraction=max_edge_flux_fraction,
            saturation_peak_fraction=saturation_peak_fraction,
            max_saturated_pixels=max_saturated_pixels,
        )
        if normalized is not None:
            stamp, weight = normalized
            stamps.append(stamp)
            weights.append(weight)
    if len(stamps) < int(min_sources):
        raise ValueError(f"Only found {len(stamps)} usable PSF source(s); require {min_sources}.")
    psf, psf_uncertainty = _stack_normalized_psf_stamps(stamps, weights)
    psf = np.clip(psf, 0.0, None)
    total = float(np.sum(psf))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Empirical PSF stack has non-positive total flux.")
    psf = psf / total
    psf_uncertainty = np.asarray(psf_uncertainty, dtype=float) / total
    if return_uncertainty:
        return psf, psf_uncertainty
    return psf


def _search_cutout_and_wcs(
    image: np.ndarray,
    wcs: WCS,
    target_pixel: tuple[float, float],
    radius: int,
) -> tuple[np.ndarray, tuple[float, float], WCS, tuple[int, int]]:
    search_image, search_target_pixel, origin = _search_cutout(image, target_pixel, radius)
    search_wcs = wcs.deepcopy()
    search_wcs.wcs.crpix -= [origin[0], origin[1]]
    return search_image, search_target_pixel, search_wcs, origin


def _search_cutout(
    image: np.ndarray,
    target_pixel: tuple[float, float],
    radius: int,
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    tx, ty = (int(round(target_pixel[0])), int(round(target_pixel[1])))
    radius = int(radius)
    x0 = max(0, tx - radius)
    x1 = min(image.shape[1], tx + radius + 1)
    y0 = max(0, ty - radius)
    y1 = min(image.shape[0], ty + radius + 1)
    search_image = np.asarray(image[y0:y1, x0:x1], dtype=float)
    search_target_pixel = (float(target_pixel[0]) - x0, float(target_pixel[1]) - y0)
    return search_image, search_target_pixel, (x0, y0)


def _candidate_ra_dec(candidate: PsfCandidate, wcs: WCS) -> tuple[float, float] | None:
    return _candidate_world_coordinates(candidate, wcs)


def _angular_distance_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (((float(ra1) - float(ra2) + 180.0) % 360.0) - 180.0) * np.cos(np.deg2rad(0.5 * (float(dec1) + float(dec2))))
    ddec = float(dec1) - float(dec2)
    return float(3600.0 * np.hypot(dra, ddec))


def _candidate_sort_key(candidate: PsfCandidate) -> tuple[bool, float, float, float]:
    scatter = float(candidate.fwhm_fractional_scatter) if np.isfinite(candidate.fwhm_fractional_scatter) else np.inf
    return bool(candidate.is_gaia_star), float(candidate.flux), -scatter, -float(candidate.size_pix)


def _match_psf_candidates_across_bands(
    candidates_by_band: dict[str, list[PsfCandidate]],
    wcs_by_band: dict[str, WCS],
    *,
    radius_arcsec: float,
) -> list[dict[str, PsfCandidate]]:
    groups: list[dict[str, PsfCandidate]] = []
    group_coords: list[tuple[float, float]] = []
    for band_code, candidates in candidates_by_band.items():
        for candidate in candidates:
            coords = _candidate_ra_dec(candidate, wcs_by_band[band_code])
            if coords is None:
                continue
            ra, dec = coords
            best_index = None
            best_distance = np.inf
            for index, (group_ra, group_dec) in enumerate(group_coords):
                if band_code in groups[index]:
                    continue
                distance = _angular_distance_arcsec(ra, dec, group_ra, group_dec)
                if distance < best_distance:
                    best_index = index
                    best_distance = distance
            if best_index is None or best_distance > float(radius_arcsec):
                groups.append({band_code: candidate})
                group_coords.append((ra, dec))
                continue
            groups[best_index][band_code] = candidate
            n = len(groups[best_index])
            group_ra, group_dec = group_coords[best_index]
            group_coords[best_index] = (group_ra + (ra - group_ra) / n, group_dec + (dec - group_dec) / n)
    groups.sort(
        key=lambda group: (
            len(group),
            any(candidate.is_gaia_star for candidate in group.values()),
            sum(float(candidate.flux) for candidate in group.values()),
        ),
        reverse=True,
    )
    return groups


def _select_common_psf_candidates(
    candidates_by_band: dict[str, list[PsfCandidate]],
    groups: list[dict[str, PsfCandidate]],
    *,
    band_codes: list[str],
    prefer_common_stars: bool,
    min_common_bands: int | None,
    max_sources: int,
    min_sources: int,
) -> dict[str, list[PsfCandidate]]:
    if not prefer_common_stars:
        return {band: list(candidates_by_band[band])[: int(max_sources)] for band in band_codes}

    required_bands = len(band_codes) if min_common_bands is None else int(min_common_bands)
    preferred_groups = [group for group in groups if len(group) >= required_bands]
    if not preferred_groups:
        preferred_groups = groups

    selected: dict[str, list[PsfCandidate]] = {band: [] for band in band_codes}
    seen: dict[str, set[tuple[float, float]]] = {band: set() for band in band_codes}
    for group in preferred_groups:
        if all(len(selected[band]) >= int(max_sources) for band in band_codes):
            break
        for band, candidate in group.items():
            if band not in selected or len(selected[band]) >= int(max_sources):
                continue
            key = (float(candidate.x_pix), float(candidate.y_pix))
            if key in seen[band]:
                continue
            selected[band].append(candidate)
            seen[band].add(key)

    for band in band_codes:
        if len(selected[band]) >= int(min_sources):
            continue
        for candidate in sorted(candidates_by_band[band], key=_candidate_sort_key, reverse=True):
            if len(selected[band]) >= int(max_sources):
                break
            key = (float(candidate.x_pix), float(candidate.y_pix))
            if key in seen[band]:
                continue
            selected[band].append(candidate)
            seen[band].add(key)
            if len(selected[band]) >= int(min_sources):
                break
    return selected


def _format_psf_candidate_diagnostics(band_code: str, diagnostics: Counter[str]) -> str:
    """Return a compact per-band PSF candidate diagnostic summary."""

    lines = [f"{band_code}: {int(diagnostics.get('detected_segments', 0))} detected segments"]
    for key in (
        "edge_flux",
        "fwhm_scatter",
        "saturated_core",
        "target_exclusion",
        "stamp_outside",
        "size",
        "neighbor",
        "nonpositive_flux",
        "peak_percentile",
        "gaia_match",
        "candidate_pool_limit",
    ):
        value = int(diagnostics.get(key, 0))
        if value:
            lines.append(f"   {value} rejected by {key}")
    lines.append(f"   {int(diagnostics.get('accepted', 0))} accepted")
    return "\n".join(lines)


def build_empirical_psfs_for_bands(
    *,
    band_specs: dict[str, str],
    target_ra_dec: tuple[float, float],
    data_dir: str | Path,
    region: str = "south",
    base_url: str = LEGACY_SURVEY_DR10_BASE_URL,
    brick: str | None = None,
    bricks_fits: str | Path | None = None,
    fit_radius: int = 45,
    config: EmpiricalPsfConfig | None = None,
) -> EmpiricalPsfResult:
    """Download/cache Legacy Survey coadds and build empirical PSFs for several bands."""

    cfg = EmpiricalPsfConfig() if config is None else config
    if cfg.psf_size % 2 == 0:
        raise ValueError("psf_size must be odd.")
    if int(cfg.psf_padding_pixels) != cfg.psf_padding_pixels or cfg.psf_padding_pixels < 0:
        raise ValueError("psf_padding_pixels must be a non-negative integer.")
    data_dir = Path(data_dir)
    if brick is None:
        bricks_path = download_legacy_survey_bricks_table(data_dir, base_url=base_url, region=region) if bricks_fits is None else Path(bricks_fits)
        brick = find_legacy_survey_brick(target_ra_dec[0], target_ra_dec[1], bricks_path)

    band_codes = list(band_specs)
    paths_by_band: dict[str, tuple[Path, Path]] = {}
    images_by_band: dict[str, np.ndarray] = {}
    headers_by_band: dict[str, Any] = {}
    search_images: dict[str, np.ndarray] = {}
    search_weights: dict[str, np.ndarray] = {}
    search_targets: dict[str, tuple[float, float]] = {}
    search_wcs_by_band: dict[str, WCS] = {}
    full_targets: dict[str, tuple[float, float]] = {}
    origins: dict[str, tuple[int, int]] = {}
    candidates_by_band: dict[str, list[PsfCandidate]] = {}
    diagnostics_by_band: dict[str, Counter[str]] = {}

    for band_code in band_codes:
        image_path, invvar_path = download_legacy_survey_coadd_band(
            data_dir,
            brick=brick,
            band=band_code,
            base_url=base_url,
            region=region,
        )
        full_image, full_header = read_legacy_survey_coadd_image(image_path)
        full_invvar, _full_invvar_header = read_legacy_survey_coadd_image(invvar_path)
        wcs = WCS(full_header, naxis=2)
        pix = wcs.all_world2pix([[target_ra_dec[0], target_ra_dec[1]]], 1)[0]
        target_pixel = (float(pix[0]), float(pix[1]))
        search_image, search_target, search_wcs, origin = _search_cutout_and_wcs(full_image, wcs, target_pixel, cfg.psf_search_radius)
        search_invvar, _search_invvar_target, _search_invvar_origin = _search_cutout(full_invvar, target_pixel, cfg.psf_search_radius)
        diagnostics: Counter[str] = Counter()
        candidates = find_empirical_psf_candidates(
            search_image,
            target_pixel=search_target,
            wcs=search_wcs if (cfg.require_gaia_match or cfg.annotate_gaia_matches) else None,
            psf_size=cfg.psf_size,
            threshold_sigma=cfg.threshold_sigma,
            npixels=cfg.npixels,
            min_sources=cfg.min_sources,
            max_sources=max(int(cfg.candidate_pool_size), int(cfg.max_sources)),
            target_exclusion_radius_pix=cfg.target_exclusion_radius_pix,
            min_size_pix=cfg.min_size_pix,
            max_size_pix=cfg.max_size_pix,
            max_peak_percentile=cfg.max_peak_percentile,
            max_edge_flux_fraction=cfg.max_edge_flux_fraction,
            saturation_peak_fraction=cfg.saturation_peak_fraction,
            max_saturated_pixels=cfg.max_saturated_pixels,
            isolation_radius_pix=cfg.isolation_radius_pix,
            max_neighbor_flux_ratio=cfg.max_neighbor_flux_ratio,
            max_fwhm_fractional_scatter=cfg.max_fwhm_fractional_scatter,
            gaia_match_radius_arcsec=cfg.gaia_match_radius_arcsec,
            gaia_xmatch_timeout=cfg.gaia_xmatch_timeout,
            require_gaia_match=cfg.require_gaia_match,
            diagnostics=diagnostics,
        )
        paths_by_band[band_code] = (image_path, invvar_path)
        images_by_band[band_code] = full_image
        headers_by_band[band_code] = full_header
        search_images[band_code] = search_image
        search_weights[band_code] = np.where(np.isfinite(search_invvar) & (search_invvar > 0.0), search_invvar, 0.0)
        search_targets[band_code] = search_target
        search_wcs_by_band[band_code] = search_wcs
        full_targets[band_code] = target_pixel
        origins[band_code] = origin
        candidates_by_band[band_code] = candidates
        diagnostics_by_band[band_code] = diagnostics
        if cfg.print_diagnostics:
            print(_format_psf_candidate_diagnostics(band_code, diagnostics))

    groups = _match_psf_candidates_across_bands(
        candidates_by_band,
        search_wcs_by_band,
        radius_arcsec=cfg.common_match_radius_arcsec,
    )
    selected_by_band = _select_common_psf_candidates(
        candidates_by_band,
        groups,
        band_codes=band_codes,
        prefer_common_stars=cfg.prefer_common_stars,
        min_common_bands=cfg.min_common_bands,
        max_sources=cfg.max_sources,
        min_sources=cfg.min_sources,
    )

    image_bands: list[ImageBandData] = []
    band_results: dict[str, EmpiricalPsfBandResult] = {}
    for band_code in band_codes:
        background = float(np.nanmedian(search_images[band_code]))
        source_image = search_images[band_code] - background
        psf, psf_uncertainty = _construct_empirical_psf_from_candidates(
            source_image,
            selected_by_band[band_code],
            target_pixel=search_targets[band_code],
            psf_size=cfg.psf_size,
            min_sources=cfg.min_sources,
            weight_image=search_weights[band_code],
            max_edge_flux_fraction=cfg.max_edge_flux_fraction,
            saturation_peak_fraction=cfg.saturation_peak_fraction,
            max_saturated_pixels=cfg.max_saturated_pixels,
            return_uncertainty=True,
        )
        image_path, invvar_path = paths_by_band[band_code]
        image_band = load_legacy_survey_coadd_band(
            image_path,
            invvar_path,
            filter_name=band_specs[band_code],
            target_ra_dec=target_ra_dec,
            radius=fit_radius,
            psf=psf,
            psf_uncertainty=psf_uncertainty,
            psf_padding_pixels=cfg.psf_padding_pixels,
        )
        image_bands.append(image_band)
        band_results[band_code] = EmpiricalPsfBandResult(
            band_code=band_code,
            filter_name=band_specs[band_code],
            image_path=image_path,
            invvar_path=invvar_path,
            psf=psf,
            psf_uncertainty=psf_uncertainty,
            candidates=candidates_by_band[band_code],
            selected_candidates=selected_by_band[band_code],
            search_image=search_images[band_code],
            search_target_pixel=search_targets[band_code],
            search_wcs=search_wcs_by_band[band_code],
            full_target_pixel=full_targets[band_code],
            search_origin=origins[band_code],
        )

    return EmpiricalPsfResult(
        brick=brick,
        image_bands=image_bands,
        bands=band_results,
        common_star_groups=groups,
        config=cfg,
    )


def download_galight_hsc_example(
    output_dir: str | Path = "data/galight_hsc",
    *,
    drive_id: str = GALIGHT_HSC_DRIVE_ID,
) -> tuple[Path, Path]:
    """Download and extract the public galight HSC QSO example data.

    The galight notebook hosts the example as a Google Drive zip. This helper
    handles Drive's large-file confirmation page and returns the HSC science and
    PSF FITS paths. Existing extracted files are reused.
    """

    output_dir = Path(output_dir)
    image_path = output_dir / GALIGHT_HSC_QSO_IMAGE
    psf_path = output_dir / GALIGHT_HSC_QSO_PSF
    if image_path.exists() and psf_path.exists():
        return image_path, psf_path

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "decomprofile_example_data.zip"
    if not zip_path.exists():
        base_url = "https://drive.google.com/uc?" + urlencode({"export": "download", "id": drive_id})
        with urlopen(base_url) as response:
            payload = response.read()
            content_type = response.headers.get("content-type", "")
        if b"Google Drive can't scan this file for viruses" in payload or "text/html" in content_type:
            html = payload.decode("utf-8", errors="replace")
            confirm = re.search(r'name="confirm" value="([^"]+)"', html)
            uuid = re.search(r'name="uuid" value="([^"]+)"', html)
            if confirm is None:
                raise RuntimeError("Could not find Google Drive download confirmation token.")
            params = {"id": drive_id, "export": "download", "confirm": confirm.group(1)}
            if uuid is not None:
                params["uuid"] = uuid.group(1)
            download_url = "https://drive.usercontent.google.com/download?" + urlencode(params)
            with urlopen(download_url) as response:
                payload = response.read()
        zip_path.write_bytes(payload)

    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    if not image_path.exists() or not psf_path.exists():
        raise FileNotFoundError("Downloaded galight archive did not contain the expected HSC QSO FITS files.")
    return image_path, psf_path


def _pixel_scale_from_header(header: Any) -> float:
    """Return approximate pixel scale in arcsec/pixel from a FITS header."""

    try:
        scale = abs(float(header["CD1_1"])) * 3600.0
        if scale > 0:
            return scale
    except Exception:
        pass
    try:
        scale = abs(float(header["CDELT1"])) * 3600.0
        if scale > 0:
            return scale
    except Exception:
        pass
    for key in ("PIXSCAL1", "PIXSCALE", "SECPIX"):
        try:
            scale = abs(float(header[key]))
            if scale > 0:
                return scale
        except Exception:
            pass
    raise ValueError("Could not infer pixel scale from FITS header.")


def _cutout(image: np.ndarray, center_xy: tuple[float, float], radius: int) -> np.ndarray:
    """Simple centered square cutout with edge padding."""

    cx, cy = center_xy
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    size = 2 * int(radius) + 1
    padded = np.pad(image, radius, mode="constant", constant_values=0.0)
    cx_p = cx_i + radius
    cy_p = cy_i + radius
    return np.asarray(padded[cy_p - radius : cy_p + radius + 1, cx_p - radius : cx_p + radius + 1], dtype=float).reshape(size, size)


def _as_2d_fits_image(data: Any) -> np.ndarray:
    image = np.asarray(data, dtype=float)
    if image.ndim > 2:
        image = np.asarray(image[0], dtype=float)
    if image.ndim != 2:
        raise ValueError("FITS image HDU must contain 2D image data.")
    return image


def _first_2d_image_hdu(hdul: fits.HDUList) -> tuple[np.ndarray, Any]:
    """Return the first HDU containing a 2D image and its header."""

    for hdu in hdul:
        if hdu.data is None:
            continue
        try:
            return _as_2d_fits_image(hdu.data), hdu.header
        except ValueError:
            continue
    raise ValueError("FITS file does not contain a 2D image HDU.")


def read_legacy_survey_coadd_image(image_fits: str | Path) -> tuple[np.ndarray, Any]:
    """Read a DR10 Legacy Survey coadd image and header."""

    with fits.open(image_fits) as hdul:
        image, header = _first_2d_image_hdu(hdul)
        return image, header.copy()


def load_hsc_band(
    image_fits: str | Path,
    psf_fits: str | Path,
    *,
    filter_name: str,
    target_ra_dec: tuple[float, float] | None = None,
    target_pixel: tuple[float, float] | None = None,
    science_hdu: int = 1,
    variance_hdu: int = 3,
    radius: int = 30,
    subtract_edge_background: bool = True,
    psf_uncertainty: np.ndarray | None = None,
    psf_padding_pixels: int = 20,
) -> ImageBandData:
    """Load an HSC/galight-style image band with an explicit PSF FITS file."""

    with fits.open(image_fits) as hdul:
        image = np.asarray(hdul[science_hdu].data, dtype=float)
        header = hdul[science_hdu].header
        primary_header = hdul[0].header
        variance = np.asarray(hdul[variance_hdu].data, dtype=float)
        noise = np.sqrt(np.clip(variance, 0.0, np.inf))
        zeropoint = 2.5 * np.log10(float(primary_header["FLUXMAG0"])) if "FLUXMAG0" in primary_header else None
        pixel_scale = _pixel_scale_from_header(header)
        if target_pixel is None:
            if target_ra_dec is None:
                raise ValueError("Provide target_ra_dec or target_pixel.")
            wcs = WCS(header, naxis=2)
            pix = wcs.all_world2pix([[target_ra_dec[0], target_ra_dec[1]]], 1)[0]
            target_pixel = (float(pix[0]), float(pix[1]))
        image_cutout = _cutout(image, target_pixel, radius)
        noise_cutout = _cutout(noise, target_pixel, radius)
        if subtract_edge_background:
            edge = np.concatenate([image_cutout[0], image_cutout[-1], image_cutout[:, 0], image_cutout[:, -1]])
            image_cutout = image_cutout - np.nanmedian(edge)

    psf = np.asarray(fits.getdata(psf_fits), dtype=float)
    band = ImageBandData(
        image=image_cutout,
        noise=np.maximum(noise_cutout, 1.0e-12),
        psf=psf,
        filter_name=filter_name,
        pixel_scale=pixel_scale,
        psf_uncertainty=None if psf_uncertainty is None else np.asarray(psf_uncertainty, dtype=float),
        zeropoint=zeropoint,
        counts_per_mjy=counts_per_mjy_from_ab_zeropoint(zeropoint) if zeropoint is not None else None,
        mask=np.ones_like(image_cutout, dtype=bool),
        header=dict(header),
        target_pixel=target_pixel,
        psf_padding_pixels=psf_padding_pixels,
    )
    band.validate()
    return band


def load_legacy_survey_coadd_band(
    image_fits: str | Path,
    invvar_fits: str | Path,
    *,
    filter_name: str,
    target_ra_dec: tuple[float, float],
    radius: int,
    psf: np.ndarray,
    psf_uncertainty: np.ndarray | None = None,
    subtract_edge_background: bool = False,
    zeropoint: float = 22.5,
    psf_padding_pixels: int = 20,
) -> ImageBandData:
    """Load a local stamp from DR10 Legacy Survey coadd image and invvar files.

    DR10 coadd images are in nanomaggies per pixel at 0.262 arcsec/pixel, and
    inverse-variance images are in 1 / nanomaggy^2.
    """

    with fits.open(image_fits) as image_hdul, fits.open(invvar_fits) as invvar_hdul:
        image, header = _first_2d_image_hdu(image_hdul)
        ivar, _ivar_header = _first_2d_image_hdu(invvar_hdul)
        pixel_scale = _pixel_scale_from_header(header)
        wcs = WCS(header, naxis=2)
        pix = wcs.all_world2pix([[target_ra_dec[0], target_ra_dec[1]]], 1)[0]
        target_pixel = (float(pix[0]), float(pix[1]))
        image_cutout = _cutout(image, target_pixel, radius)
        ivar_cutout = _cutout(ivar, target_pixel, radius)
        if subtract_edge_background:
            edge = np.concatenate([image_cutout[0], image_cutout[-1], image_cutout[:, 0], image_cutout[:, -1]])
            image_cutout = image_cutout - np.nanmedian(edge)
        noise = np.full_like(image_cutout, 1.0e30, dtype=float)
        valid = np.isfinite(ivar_cutout) & (ivar_cutout > 0.0)
        valid_mask = np.isfinite(image_cutout) & valid
        noise[valid] = 1.0 / np.sqrt(ivar_cutout[valid])

    band = ImageBandData(
        image=np.nan_to_num(image_cutout, nan=0.0, posinf=0.0, neginf=0.0),
        noise=np.maximum(noise, 1.0e-12),
        psf=np.asarray(psf, dtype=float),
        filter_name=filter_name,
        pixel_scale=pixel_scale,
        psf_uncertainty=None if psf_uncertainty is None else np.asarray(psf_uncertainty, dtype=float),
        zeropoint=zeropoint,
        counts_per_mjy=nanomaggy_counts_per_mjy() if np.isclose(float(zeropoint), 22.5) else counts_per_mjy_from_ab_zeropoint(zeropoint),
        mask=valid_mask,
        header=dict(header),
        target_pixel=target_pixel,
        psf_padding_pixels=psf_padding_pixels,
    )
    band.validate()
    return band
