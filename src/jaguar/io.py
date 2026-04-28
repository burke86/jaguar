from __future__ import annotations

from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .config import ImageBandData


GALIGHT_HSC_DRIVE_ID = "1ZO9-HzV8K60ijYWK98jGoSoZHjIGW5Lc"
GALIGHT_HSC_QSO_IMAGE = "example_data/HSC/QSO/000017.88+002612.6_HSC-I.fits"
GALIGHT_HSC_QSO_PSF = "example_data/HSC/QSO/000017.88+002612.6_HSC-I_psf.fits"
def counts_per_mjy_from_ab_zeropoint(zeropoint: float) -> float:
    """Return image counts per mJy for an AB zeropoint."""

    return float(10 ** ((float(zeropoint) - 16.4) / 2.5))


def counts_to_mjy(counts: float, counts_per_mjy: float) -> float:
    """Convert image counts to mJy."""

    return float(counts) / float(counts_per_mjy)


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
            wcs = WCS(header)
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
        zeropoint=zeropoint,
        counts_per_mjy=counts_per_mjy_from_ab_zeropoint(zeropoint) if zeropoint is not None else None,
        mask=np.ones_like(image_cutout, dtype=bool),
        header=dict(header),
        target_pixel=target_pixel,
    )
    band.validate()
    return band
