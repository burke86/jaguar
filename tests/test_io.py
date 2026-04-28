from __future__ import annotations

import numpy as np
from astropy.io import fits

from jaguar.io import load_hsc_band


def test_load_hsc_band_from_fits(tmp_path):
    image = np.ones((25, 25), dtype=float) * 5.0
    variance = np.ones_like(image) * 4.0
    psf = np.ones((5, 5), dtype=float)
    primary = fits.PrimaryHDU()
    primary.header["FLUXMAG0"] = 10 ** 12
    sci = fits.ImageHDU(image)
    sci.header["CDELT1"] = -0.168 / 3600.0
    sci.header["CDELT2"] = 0.168 / 3600.0
    hdul = fits.HDUList([primary, sci, fits.ImageHDU(np.zeros_like(image)), fits.ImageHDU(variance)])
    image_path = tmp_path / "image.fits"
    psf_path = tmp_path / "psf.fits"
    hdul.writeto(image_path)
    fits.PrimaryHDU(psf).writeto(psf_path)

    band = load_hsc_band(
        image_path,
        psf_path,
        filter_name="hsc_i",
        target_pixel=(12, 12),
        radius=5,
        subtract_edge_background=False,
    )
    assert band.image.shape == (11, 11)
    assert band.noise.shape == (11, 11)
    assert band.psf.shape == (5, 5)
    assert np.isclose(band.pixel_scale, 0.168)
    assert np.isclose(band.zeropoint, 30.0)

