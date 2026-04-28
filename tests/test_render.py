from __future__ import annotations

import numpy as np

from jaguar.render import psf_unit_flux, sersic_ellipse_unit_flux


def test_psf_normalization_preserves_flux():
    psf = np.ones((5, 5))
    image = np.asarray(psf_unit_flux(psf, (21, 21)))
    assert np.isfinite(image).all()
    assert np.isclose(image.sum(), 1.0)


def test_sersic_normalization_preserves_flux():
    image = np.asarray(
        sersic_ellipse_unit_flux(
            (41, 41),
            0.168,
            reff_arcsec=0.5,
            n_sersic=2.0,
            e1=0.1,
            e2=0.0,
        )
    )
    assert np.isfinite(image).all()
    assert np.isclose(image.sum(), 1.0)
    assert image.max() > image.mean()

