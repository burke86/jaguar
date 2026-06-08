from __future__ import annotations

import numpy as np

from jaguar.render import (
    convolve_fft_same,
    convolve_fft_same_precomputed,
    pad_psf,
    pixel_coordinates,
    prepare_fft_kernel,
    psf_unit_flux,
    psf_unit_flux_uncertainty,
    sersic_ellipse_unit_flux,
)


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


def test_precomputed_fft_convolution_matches_direct_convolution():
    image = np.zeros((17, 19), dtype=float)
    image[8, 9] = 1.0
    image[6, 12] = 0.2
    kernel = np.ones((5, 7), dtype=float)

    direct = np.asarray(convolve_fft_same(image, kernel))
    prepared = prepare_fft_kernel(kernel, image.shape)
    precomputed = np.asarray(convolve_fft_same_precomputed(image, *prepared))

    assert np.allclose(precomputed, direct)
    assert np.isclose(precomputed.sum(), 1.0)


def test_pixel_coordinate_grids_are_cached():
    xx1, yy1 = pixel_coordinates((11, 13), 0.2)
    xx2, yy2 = pixel_coordinates((11, 13), 0.2)

    assert xx1 is xx2
    assert yy1 is yy2
