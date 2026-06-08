from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchedFilterResult:
    """Compressed point-source likelihood from one image stamp.

    The model is linear in source flux and constant background:

        data = flux * psf + background + noise

    The returned flux and uncertainty are the weighted least-squares
    conditional estimates after fitting the constant background term.
    """

    flux: float
    flux_err: float
    background: float
    background_err: float
    reduced_chi2: float
    npixels: int


@dataclass(frozen=True)
class BatchMatchedFilterResult:
    """Batched compressed point-source likelihoods."""

    flux: np.ndarray
    flux_err: np.ndarray
    background: np.ndarray
    background_err: np.ndarray
    reduced_chi2: np.ndarray
    npixels: np.ndarray


def moffat_alpha_from_fwhm(fwhm_pix: float, beta: float) -> float:
    """Return Moffat alpha in pixels for a target FWHM and beta."""

    fwhm = float(fwhm_pix)
    beta = float(beta)
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        raise ValueError("fwhm_pix must be positive and finite.")
    if not np.isfinite(beta) or beta <= 1.0:
        raise ValueError("beta must be finite and greater than 1.")
    return fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def moffat_unit_flux(
    shape: tuple[int, int],
    *,
    fwhm_pix: float,
    beta: float = 3.5,
    dx_pix: float = 0.0,
    dy_pix: float = 0.0,
    q: float = 1.0,
    theta_rad: float = 0.0,
) -> np.ndarray:
    """Render an elliptical Moffat PSF normalized to unit total flux."""

    ny, nx = int(shape[0]), int(shape[1])
    if ny <= 0 or nx <= 0:
        raise ValueError("shape must contain positive dimensions.")
    q = float(q)
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("q must be positive and finite.")
    alpha = moffat_alpha_from_fwhm(fwhm_pix, beta)
    y, x = np.indices((ny, nx), dtype=float)
    x = x - (nx - 1) / 2.0 - float(dx_pix)
    y = y - (ny - 1) / 2.0 - float(dy_pix)
    cos_t = np.cos(float(theta_rad))
    sin_t = np.sin(float(theta_rad))
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    r2 = x_rot**2 + (y_rot / q) ** 2
    image = (1.0 + r2 / alpha**2) ** (-float(beta))
    image = np.clip(image, 0.0, np.inf)
    total = float(np.sum(image))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Moffat image has non-positive total flux.")
    return image / total


def moffat_unit_flux_batch(
    shape: tuple[int, int],
    *,
    fwhm_pix: float,
    beta: float = 3.5,
    dx_pix: np.ndarray | float = 0.0,
    dy_pix: np.ndarray | float = 0.0,
    q: float = 1.0,
    theta_rad: float = 0.0,
) -> np.ndarray:
    """Render a batch of unit-flux elliptical Moffat PSFs.

    ``dx_pix`` and ``dy_pix`` are broadcast to a common one-dimensional shape.
    The return value has shape ``(n, ny, nx)``.
    """

    dx = np.atleast_1d(np.asarray(dx_pix, dtype=float))
    dy = np.atleast_1d(np.asarray(dy_pix, dtype=float))
    dx, dy = np.broadcast_arrays(dx, dy)
    ny, nx = int(shape[0]), int(shape[1])
    if ny <= 0 or nx <= 0:
        raise ValueError("shape must contain positive dimensions.")
    q = float(q)
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("q must be positive and finite.")
    alpha = moffat_alpha_from_fwhm(fwhm_pix, beta)
    y, x = np.indices((ny, nx), dtype=float)
    x = x[None, :, :] - (nx - 1) / 2.0 - dx[:, None, None]
    y = y[None, :, :] - (ny - 1) / 2.0 - dy[:, None, None]
    cos_t = np.cos(float(theta_rad))
    sin_t = np.sin(float(theta_rad))
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    r2 = x_rot**2 + (y_rot / q) ** 2
    images = np.clip((1.0 + r2 / alpha**2) ** (-float(beta)), 0.0, np.inf)
    totals = np.sum(images, axis=(1, 2), keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("Moffat image has non-positive total flux.")
    return images / totals


def matched_filter_flux(
    data: np.ndarray,
    noise: float | np.ndarray,
    psf: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    fit_background: bool = True,
) -> MatchedFilterResult:
    """Compress one stamp to a weighted point-source flux likelihood."""

    data = np.asarray(data, dtype=float)
    noise = np.asarray(noise, dtype=float)
    psf = np.asarray(psf, dtype=float)
    scalar_noise = noise.ndim == 0
    if data.shape != psf.shape:
        raise ValueError("data and psf must have matching shapes.")
    if not scalar_noise and data.shape != noise.shape:
        raise ValueError("noise must be scalar or match data shape.")
    if scalar_noise:
        noise_value = float(noise)
        valid = np.isfinite(data) & np.isfinite(psf) & np.isfinite(noise_value) & (noise_value > 0.0)
    else:
        valid = np.isfinite(data) & np.isfinite(noise) & np.isfinite(psf) & (noise > 0.0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if int(np.sum(valid)) < (2 if fit_background else 1):
        return MatchedFilterResult(np.nan, np.nan, np.nan, np.nan, np.nan, int(np.sum(valid)))

    y = data[valid]
    p = psf[valid]
    if scalar_noise:
        w = np.ones_like(p) / noise_value**2
    else:
        w = 1.0 / noise[valid] ** 2
    if fit_background:
        one = np.ones_like(p)
        a00 = float(np.sum(w * p * p))
        a01 = float(np.sum(w * p * one))
        a11 = float(np.sum(w * one * one))
        b0 = float(np.sum(w * p * y))
        b1 = float(np.sum(w * one * y))
        design = np.asarray([[a00, a01], [a01, a11]], dtype=float)
        rhs = np.asarray([b0, b1], dtype=float)
        try:
            cov = np.linalg.inv(design)
            flux, background = cov @ rhs
        except np.linalg.LinAlgError:
            return MatchedFilterResult(np.nan, np.nan, np.nan, np.nan, np.nan, int(np.sum(valid)))
        model = flux * p + background
        flux_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        background_err = float(np.sqrt(max(cov[1, 1], 0.0)))
        dof = max(int(np.sum(valid)) - 2, 1)
    else:
        denom = float(np.sum(w * p * p))
        if denom <= 0.0:
            return MatchedFilterResult(np.nan, np.nan, np.nan, np.nan, np.nan, int(np.sum(valid)))
        flux = float(np.sum(w * p * y) / denom)
        background = 0.0
        model = flux * p
        flux_err = float(1.0 / np.sqrt(denom))
        background_err = np.nan
        dof = max(int(np.sum(valid)) - 1, 1)

    residual = y - model
    reduced_chi2 = float(np.sum(w * residual**2) / dof)
    return MatchedFilterResult(
        flux=float(flux),
        flux_err=flux_err,
        background=float(background),
        background_err=background_err,
        reduced_chi2=reduced_chi2,
        npixels=int(np.sum(valid)),
    )


def matched_filter_flux_batch(
    data: np.ndarray,
    noise: float | np.ndarray,
    psf: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    fit_background: bool = True,
) -> BatchMatchedFilterResult:
    """Compress a batch of stamps to weighted point-source flux likelihoods."""

    data = np.asarray(data, dtype=float)
    psf = np.asarray(psf, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if data.ndim != 3 or psf.shape != data.shape:
        raise ValueError("data and psf must have matching shape (n, ny, nx).")
    scalar_noise = noise.ndim == 0
    if scalar_noise:
        noise_value = float(noise)
        valid = np.isfinite(data) & np.isfinite(psf) & np.isfinite(noise_value) & (noise_value > 0.0)
        w = np.ones_like(data) / noise_value**2
    else:
        if noise.shape == data.shape[1:]:
            noise = np.broadcast_to(noise[None, :, :], data.shape)
        elif noise.shape != data.shape:
            raise ValueError("noise must be scalar, stamp-shaped, or match data batch shape.")
        valid = np.isfinite(data) & np.isfinite(noise) & np.isfinite(psf) & (noise > 0.0)
        w = np.where(valid, 1.0 / np.maximum(noise, 1.0e-300) ** 2, 0.0)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape == data.shape[1:]:
            mask_array = np.broadcast_to(mask_array[None, :, :], data.shape)
        elif mask_array.shape != data.shape:
            raise ValueError("mask must be stamp-shaped or match data batch shape.")
        valid &= mask_array
    w = np.where(valid, w, 0.0)
    npixels = np.sum(valid, axis=(1, 2)).astype(int)
    y = np.where(valid, data, 0.0)
    p = np.where(valid, psf, 0.0)

    if fit_background:
        a00 = np.sum(w * p * p, axis=(1, 2))
        a01 = np.sum(w * p, axis=(1, 2))
        a11 = np.sum(w, axis=(1, 2))
        b0 = np.sum(w * p * y, axis=(1, 2))
        b1 = np.sum(w * y, axis=(1, 2))
        det = a00 * a11 - a01 * a01
        good = (npixels >= 2) & np.isfinite(det) & (np.abs(det) > 0.0)
        flux = np.full(data.shape[0], np.nan)
        background = np.full(data.shape[0], np.nan)
        flux[good] = (a11[good] * b0[good] - a01[good] * b1[good]) / det[good]
        background[good] = (-a01[good] * b0[good] + a00[good] * b1[good]) / det[good]
        flux_err = np.full(data.shape[0], np.nan)
        background_err = np.full(data.shape[0], np.nan)
        flux_err[good] = np.sqrt(np.maximum(a11[good] / det[good], 0.0))
        background_err[good] = np.sqrt(np.maximum(a00[good] / det[good], 0.0))
        model = flux[:, None, None] * p + background[:, None, None]
        dof = np.maximum(npixels - 2, 1)
    else:
        denom = np.sum(w * p * p, axis=(1, 2))
        good = (npixels >= 1) & np.isfinite(denom) & (denom > 0.0)
        flux = np.full(data.shape[0], np.nan)
        flux[good] = np.sum(w[good] * p[good] * y[good], axis=(1, 2)) / denom[good]
        background = np.zeros(data.shape[0])
        flux_err = np.full(data.shape[0], np.nan)
        flux_err[good] = 1.0 / np.sqrt(denom[good])
        background_err = np.full(data.shape[0], np.nan)
        model = flux[:, None, None] * p
        dof = np.maximum(npixels - 1, 1)

    residual = np.where(valid, y - model, 0.0)
    reduced_chi2 = np.sum(w * residual**2, axis=(1, 2)) / dof
    reduced_chi2 = np.where(good, reduced_chi2, np.nan)
    return BatchMatchedFilterResult(
        flux=flux,
        flux_err=flux_err,
        background=background,
        background_err=background_err,
        reduced_chi2=reduced_chi2,
        npixels=npixels,
    )
