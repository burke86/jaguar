from __future__ import annotations

import numpy as np

from jaguar.render import pad_psf, psf_unit_flux, psf_unit_flux_uncertainty, sersic_ellipse_unit_flux


def test_psf_normalization_preserves_flux():
    psf = np.ones((5, 5))
    image = np.asarray(psf_unit_flux(psf, (21, 21)))
    assert np.isfinite(image).all()
    assert np.isclose(image.sum(), 1.0)


def test_pad_psf_zero_pads_without_inventing_edge_flux():
    psf = np.ones((5, 5), dtype=float)
    padded = np.asarray(pad_psf(psf, padding_pixels=4))
    assert padded.shape == (13, 13)
    assert np.isclose(padded.sum(), 1.0)
    assert np.allclose(padded[0, :], 0.0)
    assert padded[3, 6] == 0.0
    assert padded[4, 6] > 0.0


def test_psf_unit_flux_accepts_padding_pixels():
    psf = np.ones((5, 5), dtype=float)
    image = np.asarray(psf_unit_flux(psf, (31, 31), padding_pixels=4))
    assert np.isfinite(image).all()
    assert np.isclose(image.sum(), 1.0)
    assert image[15, 15] > 0.0
    assert image[10, 15] == 0.0
    assert image[10 + 4, 15] > 0.0


def test_psf_unit_flux_uncertainty_is_not_renormalized():
    uncertainty = np.ones((5, 5), dtype=float) * 0.01
    image = np.asarray(psf_unit_flux_uncertainty(uncertainty, (21, 21), padding_pixels=2))
    assert np.isfinite(image).all()
    assert np.isclose(image.max(), 0.01)
    assert np.isclose(image.sum(), uncertainty.sum())


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
