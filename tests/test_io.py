from __future__ import annotations

import sys
import types

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jaguar.io import (
    EmpiricalPsfBandResult,
    EmpiricalPsfConfig,
    EmpiricalPsfResult,
    ImageBandData,
    PsfCandidate,
    _construct_empirical_psf_from_candidates,
    build_empirical_psfs_for_bands,
    build_empirical_psfs_for_galex_bands,
    combine_empirical_psf_results,
    construct_empirical_psf,
    download_galex_coadd_blocks,
    find_legacy_survey_brick,
    find_empirical_psf_candidates,
    legacy_survey_coadd_url,
    load_hsc_band,
    load_legacy_survey_coadd_band,
    nanomaggy_counts_per_mjy,
    query_galex_coadd_products,
    read_legacy_survey_coadd_image,
)


def _gaussian(shape, x0, y0, sigma, amp):
    yy, xx = np.indices(shape, dtype=float)
    return amp * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sigma**2)


def _test_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [0.0, 0.0]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cdelt = [-0.2 / 3600.0, 0.2 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _install_fake_xmatch(monkeypatch, query):
    astroquery_module = types.ModuleType("astroquery")
    xmatch_module = types.ModuleType("astroquery.xmatch")
    xmatch_module.XMatch = types.SimpleNamespace(query=query)
    monkeypatch.setitem(sys.modules, "astroquery", astroquery_module)
    monkeypatch.setitem(sys.modules, "astroquery.xmatch", xmatch_module)


def _install_fake_mast_observations(monkeypatch, observations):
    astroquery_module = types.ModuleType("astroquery")
    mast_module = types.ModuleType("astroquery.mast")
    mast_module.Observations = observations
    monkeypatch.setitem(sys.modules, "astroquery", astroquery_module)
    monkeypatch.setitem(sys.modules, "astroquery.mast", mast_module)


def _write_legacy_test_coadds(data_dir, brick, band, image, ivar, header):
    path = data_dir / "dr10" / "south" / "coadd" / brick[:3] / brick
    path.mkdir(parents=True, exist_ok=True)
    image_path = path / f"legacysurvey-{brick}-image-{band}.fits.fz"
    invvar_path = path / f"legacysurvey-{brick}-invvar-{band}.fits.fz"
    fits.PrimaryHDU(image, header=header).writeto(image_path)
    fits.PrimaryHDU(ivar, header=header).writeto(invvar_path)
    return image_path, invvar_path


def _legacy_test_wcs_header(shape=(101, 101)):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cdelt = [-0.262 / 3600.0, 0.262 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs, wcs.to_header()


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


def test_legacy_survey_coadd_url_uses_dr10_static_paths():
    url = legacy_survey_coadd_url("0000p004", "i")
    assert "dr10/south/coadd/000/0000p004" in url
    assert url.endswith("legacysurvey-0000p004-image-i.fits.fz")
    assert "fits-cutout" not in url


def test_query_galex_coadd_products_returns_main_intensity_maps(monkeypatch):
    class FakeObservations:
        @staticmethod
        def query_criteria(**kwargs):
            assert kwargs["obs_collection"] == "GALEX"
            assert kwargs["dataproduct_type"] == "image"
            return Table(rows=[("obs",)], names=("obsid",))

        @staticmethod
        def get_product_list(_observations):
            return Table(
                rows=[
                    ("FUV", "SCIENCE", "AIS_326_sg34-fd-int.fits.gz", "mast:GALEX/url/data/GR6/pipe/AIS_326/d/01-main/0001-img/07-try/AIS_326_sg34-fd-int.fits.gz", "1"),
                    ("FUV", "SCIENCE", "AIS_326_0001_sg34-fd-int.fits.gz", "mast:GALEX/url/data/GR6/pipe/AIS_326/d/00-visits/0001-img/07-try/AIS_326_0001_sg34-fd-int.fits.gz", "1"),
                    ("NUV", "SCIENCE", "AIS_326_sg34-nd-int.fits.gz", "mast:GALEX/url/data/GR6/pipe/AIS_326/d/01-main/0001-img/07-try/AIS_326_sg34-nd-int.fits.gz", "1"),
                    ("NUV", "SCIENCE", "AIS_326_sg34-nd-cnt.fits.gz", "mast:GALEX/url/data/GR6/pipe/AIS_326/d/01-main/0001-img/07-try/AIS_326_sg34-nd-cnt.fits.gz", "1"),
                    ("NUV", "PREVIEW", "AIS_326_sg34-nd-int.jpg", "mast:GALEX/url/data/GR6/pipe/AIS_326/d/01-main/0001-img/07-try/AIS_326_sg34-nd-int.jpg", "1"),
                ],
                names=("filters", "productType", "productFilename", "dataURI", "obsID"),
            )

    _install_fake_mast_observations(monkeypatch, FakeObservations)

    products = query_galex_coadd_products((180.737219, -20.934155))

    assert [product.product_filename for product in products["FUV"]] == ["AIS_326_sg34-fd-int.fits.gz"]
    assert [product.product_filename for product in products["NUV"]] == ["AIS_326_sg34-nd-int.fits.gz"]


def test_download_galex_coadd_blocks_selects_wcs_containing_target(monkeypatch, tmp_path):
    target_ra_dec = (180.737219, -20.934155)
    calls = []

    def write_image(path, crval):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [10.0, 10.0]
        wcs.wcs.crval = list(crval)
        wcs.wcs.cdelt = [-1.5 / 3600.0, 1.5 / 3600.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fits.PrimaryHDU(np.ones((21, 21), dtype=float), header=wcs.to_header()).writeto(path, overwrite=True)

    class FakeObservations:
        @staticmethod
        def query_criteria(**_kwargs):
            return Table(rows=[("obs",)], names=("obsid",))

        @staticmethod
        def get_product_list(_observations):
            return Table(
                rows=[
                    ("FUV", "SCIENCE", "fuv-hit-fd-int.fits.gz", "mast:GALEX/url/data/d/01-main/fuv-hit", "1"),
                    ("NUV", "SCIENCE", "nuv-a-miss-nd-int.fits.gz", "mast:GALEX/url/data/d/01-main/nuv-miss", "1"),
                    ("NUV", "SCIENCE", "nuv-hit-nd-int.fits.gz", "mast:GALEX/url/data/d/01-main/nuv-hit", "1"),
                ],
                names=("filters", "productType", "productFilename", "dataURI", "obsID"),
            )

        @staticmethod
        def download_file(uri, *, local_path=None, **_kwargs):
            calls.append(uri)
            crval = (10.0, 10.0) if uri.endswith("nuv-miss") else target_ra_dec
            write_image(local_path, crval)

    _install_fake_mast_observations(monkeypatch, FakeObservations)

    paths = download_galex_coadd_blocks(tmp_path, target_ra_dec)

    assert paths["FUV"].name == "fuv-hit-fd-int.fits.gz"
    assert paths["NUV"].name == "nuv-hit-nd-int.fits.gz"
    assert calls == [
        "mast:GALEX/url/data/d/01-main/fuv-hit",
        "mast:GALEX/url/data/d/01-main/nuv-miss",
        "mast:GALEX/url/data/d/01-main/nuv-hit",
    ]


def test_download_galex_coadd_blocks_uses_cached_files_before_mast(monkeypatch, tmp_path):
    target_ra_dec = (180.737219, -20.934155)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.0, 10.0]
    wcs.wcs.crval = list(target_ra_dec)
    wcs.wcs.cdelt = [-1.5 / 3600.0, 1.5 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    fits.PrimaryHDU(np.ones((21, 21), dtype=float), header=wcs.to_header()).writeto(tmp_path / "cached-fd-int.fits.gz")
    fits.PrimaryHDU(np.ones((21, 21), dtype=float), header=wcs.to_header()).writeto(tmp_path / "cached-nd-int.fits.gz")

    class FakeObservations:
        @staticmethod
        def query_criteria(**_kwargs):
            raise AssertionError("MAST should not be queried when cached GALEX blocks contain the target")

    _install_fake_mast_observations(monkeypatch, FakeObservations)

    paths = download_galex_coadd_blocks(tmp_path, target_ra_dec)

    assert paths["FUV"].name == "cached-fd-int.fits.gz"
    assert paths["NUV"].name == "cached-nd-int.fits.gz"


def test_find_legacy_survey_brick_from_summary_table(tmp_path):
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="brickname", format="8A", array=np.asarray(["0000p004"])),
            fits.Column(name="ra", format="D", array=np.asarray([0.05])),
            fits.Column(name="dec", format="D", array=np.asarray([0.45])),
            fits.Column(name="ra1", format="D", array=np.asarray([0.0])),
            fits.Column(name="ra2", format="D", array=np.asarray([0.1])),
            fits.Column(name="dec1", format="D", array=np.asarray([0.4])),
            fits.Column(name="dec2", format="D", array=np.asarray([0.5])),
        ]
    )
    path = tmp_path / "survey-bricks.fits"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    assert find_legacy_survey_brick(0.0745, 0.4368, path) == "0000p004"


def test_construct_empirical_psf_uses_bright_compact_sources():
    shape = (81, 81)
    image = np.ones(shape, dtype=float) * 2.0
    image += _gaussian(shape, 15, 15, 1.2, 100.0)
    image += _gaussian(shape, 65, 20, 1.2, 80.0)
    image += _gaussian(shape, 40, 40, 1.0, 200.0)

    psf = construct_empirical_psf(
        image,
        target_pixel=(40, 40),
        psf_size=15,
        threshold_sigma=5.0,
        target_exclusion_radius_pix=10.0,
        min_sources=2,
    )

    assert psf.shape == (15, 15)
    assert np.isclose(np.sum(psf), 1.0)
    assert psf[7, 7] == np.max(psf)


def test_construct_empirical_psf_subtracts_local_annulus_background():
    shape = (61, 61)
    image = np.ones(shape, dtype=float) * 2.0
    image += _gaussian(shape, 30, 30, 1.2, 100.0)
    image[21:40, 21:40] += 5.0
    candidate = PsfCandidate(30.0, 30.0, 100.0, 1.0, 100.0)

    psf = _construct_empirical_psf_from_candidates(
        image,
        [candidate],
        target_pixel=(30.0, 30.0),
        psf_size=11,
        min_sources=1,
    )

    edge = np.concatenate([psf[0], psf[-1], psf[:, 0], psf[:, -1]])
    assert np.nanmax(edge) < 1.0e-4
    assert np.isclose(np.sum(psf), 1.0)


def test_construct_empirical_psf_uses_inverse_variance_weighting():
    shape = (81, 81)
    clean = _gaussian(shape, 20, 20, 1.0, 100.0)
    noisy = _gaussian(shape, 60, 20, 1.0, 100.0)
    noisy[20, 62] += 100.0
    image = clean + noisy
    weights = np.ones(shape, dtype=float)
    weights[15:26, 55:66] = 1.0e-4
    candidates = [
        PsfCandidate(20.0, 20.0, 100.0, 1.0, 100.0),
        PsfCandidate(60.0, 20.0, 100.0, 1.0, 100.0),
    ]

    weighted = _construct_empirical_psf_from_candidates(
        image,
        candidates,
        target_pixel=(40.0, 40.0),
        psf_size=11,
        min_sources=2,
        weight_image=weights,
    )
    unweighted = _construct_empirical_psf_from_candidates(
        image,
        candidates,
        target_pixel=(40.0, 40.0),
        psf_size=11,
        min_sources=2,
    )

    assert weighted[5, 7] < unweighted[5, 7]
    assert np.isclose(np.sum(weighted), 1.0)


def test_construct_empirical_psf_rejects_edge_flux_dominated_stamps():
    shape = (81, 81)
    image = _gaussian(shape, 20, 20, 6.0, 100.0)
    image += _gaussian(shape, 60, 20, 1.0, 100.0)
    candidates = [
        PsfCandidate(20.0, 20.0, 100.0, 1.0, 100.0),
        PsfCandidate(60.0, 20.0, 100.0, 1.0, 100.0),
    ]

    psf = _construct_empirical_psf_from_candidates(
        image,
        candidates,
        target_pixel=(40.0, 40.0),
        psf_size=11,
        min_sources=1,
        max_edge_flux_fraction=0.05,
    )

    edge = np.concatenate([psf[0], psf[-1], psf[1:-1, 0], psf[1:-1, -1]])
    assert np.sum(edge) / np.sum(psf) < 0.05
    assert psf[5, 5] == np.max(psf)


def test_construct_empirical_psf_rejects_saturated_core_stamps():
    shape = (81, 81)
    saturated = _gaussian(shape, 20, 20, 1.0, 100.0)
    saturated[19:22, 19:22] = np.max(saturated)
    compact = _gaussian(shape, 60, 20, 1.0, 80.0)
    image = saturated + compact
    candidates = [
        PsfCandidate(20.0, 20.0, 100.0, 1.0, 100.0),
        PsfCandidate(60.0, 20.0, 80.0, 1.0, 80.0),
    ]

    psf = _construct_empirical_psf_from_candidates(
        image,
        candidates,
        target_pixel=(40.0, 40.0),
        psf_size=11,
        min_sources=1,
        max_saturated_pixels=4,
    )

    assert psf[5, 5] == np.max(psf)
    assert np.count_nonzero(psf >= 0.95 * np.max(psf)) <= 4


def test_construct_empirical_psf_returns_stack_uncertainty():
    shape = (81, 81)
    image = _gaussian(shape, 20, 20, 1.0, 100.0)
    image += _gaussian(shape, 60, 20, 1.1, 90.0)
    weights = np.ones(shape, dtype=float) * 4.0
    candidates = [
        PsfCandidate(20.0, 20.0, 100.0, 1.0, 100.0),
        PsfCandidate(60.0, 20.0, 90.0, 1.0, 90.0),
    ]

    psf, uncertainty = _construct_empirical_psf_from_candidates(
        image,
        candidates,
        target_pixel=(40.0, 40.0),
        psf_size=11,
        min_sources=2,
        weight_image=weights,
        return_uncertainty=True,
    )

    assert psf.shape == (11, 11)
    assert uncertainty.shape == psf.shape
    assert np.isclose(np.sum(psf), 1.0)
    assert np.all(np.isfinite(uncertainty))
    assert np.all(uncertainty >= 0.0)
    assert np.max(uncertainty) > 0.0


def test_find_empirical_psf_candidates_matches_psf_selection():
    shape = (81, 81)
    image = np.ones(shape, dtype=float) * 2.0
    image += _gaussian(shape, 15, 15, 1.2, 100.0)
    image += _gaussian(shape, 65, 20, 1.2, 80.0)
    image += _gaussian(shape, 40, 40, 1.0, 200.0)

    candidates = find_empirical_psf_candidates(
        image,
        target_pixel=(40, 40),
        psf_size=15,
        threshold_sigma=5.0,
        target_exclusion_radius_pix=10.0,
        min_sources=2,
    )

    assert len(candidates) == 2
    assert candidates[0].flux > candidates[1].flux
    assert all(np.hypot(candidate.x_pix - 40.0, candidate.y_pix - 40.0) > 10.0 for candidate in candidates)


def test_find_empirical_psf_candidates_marks_and_requires_gaia_matches(monkeypatch):
    def query(*, cat1, cat2, max_distance, colRA1, colDec1):
        from astroquery.xmatch import XMatch

        assert cat2 == "vizier:I/355/gaiadr3"
        assert colRA1 == "ra"
        assert colDec1 == "dec"
        assert np.isclose(max_distance.to_value("arcsec"), 1.0)
        assert np.isclose(XMatch.TIMEOUT, 3.0)
        assert len(cat1) == 2
        return Table(rows=[(0,)], names=("candidate_index",))

    _install_fake_xmatch(monkeypatch, query)
    shape = (81, 81)
    image = np.ones(shape, dtype=float) * 2.0
    image += _gaussian(shape, 15, 15, 1.2, 100.0)
    image += _gaussian(shape, 65, 20, 1.2, 80.0)
    image += _gaussian(shape, 40, 40, 1.0, 200.0)

    candidates = find_empirical_psf_candidates(
        image,
        target_pixel=(40, 40),
        wcs=_test_wcs(),
        psf_size=15,
        threshold_sigma=5.0,
        target_exclusion_radius_pix=10.0,
        min_sources=2,
        gaia_xmatch_timeout=3.0,
        require_gaia_match=True,
    )

    assert len(candidates) == 1
    assert candidates[0].is_gaia_star
    assert np.isclose(candidates[0].x_pix, 15.0, atol=1.0)


def test_construct_empirical_psf_skips_gaia_xmatch_unless_required(monkeypatch):
    def query(**_kwargs):
        raise AssertionError("XMatch should not run unless Gaia matching is required for the PSF stack")

    _install_fake_xmatch(monkeypatch, query)
    shape = (81, 81)
    image = np.ones(shape, dtype=float) * 2.0
    image += _gaussian(shape, 15, 15, 1.2, 100.0)
    image += _gaussian(shape, 65, 20, 1.2, 80.0)
    image += _gaussian(shape, 40, 40, 1.0, 200.0)

    psf = construct_empirical_psf(
        image,
        target_pixel=(40, 40),
        wcs=_test_wcs(),
        psf_size=15,
        threshold_sigma=5.0,
        target_exclusion_radius_pix=10.0,
        min_sources=2,
        require_gaia_match=False,
    )

    assert psf.shape == (15, 15)
    assert np.isclose(np.sum(psf), 1.0)


def test_find_empirical_psf_candidates_continues_when_gaia_xmatch_fails(monkeypatch):
    def query(**_kwargs):
        raise TimeoutError("CDS XMatch is unavailable")

    _install_fake_xmatch(monkeypatch, query)
    shape = (81, 81)
    image = np.ones(shape, dtype=float) * 2.0
    image += _gaussian(shape, 15, 15, 1.2, 100.0)
    image += _gaussian(shape, 65, 20, 1.2, 80.0)
    image += _gaussian(shape, 40, 40, 1.0, 200.0)

    candidates = find_empirical_psf_candidates(
        image,
        target_pixel=(40, 40),
        wcs=_test_wcs(),
        psf_size=15,
        threshold_sigma=5.0,
        target_exclusion_radius_pix=10.0,
        min_sources=2,
        require_gaia_match=True,
    )

    assert len(candidates) == 2
    assert not any(candidate.is_gaia_star for candidate in candidates)


def test_find_empirical_psf_candidates_rejects_bright_neighbors_in_stamp():
    shape = (101, 101)
    image = np.ones(shape, dtype=float) * 2.0
    image += _gaussian(shape, 20, 20, 1.2, 100.0)
    image += _gaussian(shape, 27, 20, 1.2, 30.0)
    image += _gaussian(shape, 80, 20, 1.2, 70.0)

    candidates = find_empirical_psf_candidates(
        image,
        target_pixel=(50, 50),
        psf_size=21,
        threshold_sigma=5.0,
        target_exclusion_radius_pix=10.0,
        min_sources=1,
        isolation_radius_pix=11.0,
        max_neighbor_flux_ratio=0.05,
    )

    assert len(candidates) == 1
    assert np.isclose(candidates[0].x_pix, 80.0, atol=1.0)


def test_find_empirical_psf_candidates_rejects_fwhm_asymmetric_sources():
    shape = (101, 101)
    image = np.ones(shape, dtype=float) * 2.0
    image += _gaussian(shape, 25, 25, 1.2, 80.0)
    yy, xx = np.indices(shape, dtype=float)
    image += 100.0 * np.exp(-0.5 * (((xx - 75.0) / 1.0) ** 2 + ((yy - 25.0) / 3.0) ** 2))

    candidates = find_empirical_psf_candidates(
        image,
        target_pixel=(50, 50),
        psf_size=21,
        threshold_sigma=5.0,
        target_exclusion_radius_pix=10.0,
        min_sources=1,
        max_fwhm_fractional_scatter=0.1,
    )

    assert len(candidates) == 1
    assert np.isclose(candidates[0].x_pix, 25.0, atol=1.0)
    assert candidates[0].fwhm_fractional_scatter <= 0.1


def test_construct_empirical_psf_fails_without_usable_psf_stars():
    shape = (41, 41)
    image = np.ones(shape, dtype=float) * 2.0 + _gaussian(shape, 20, 20, 1.2, 100.0)

    try:
        construct_empirical_psf(
            image,
            target_pixel=(20, 20),
            psf_size=15,
            threshold_sigma=20.0,
            target_exclusion_radius_pix=10.0,
        )
    except ValueError as exc:
        assert "No compact sources" in str(exc)
    else:
        raise AssertionError("Expected missing PSF stars to raise ValueError.")


def test_load_legacy_survey_coadd_band_uses_inverse_variance(tmp_path):
    image = np.ones((31, 31), dtype=float) * 5.0
    ivar = np.ones_like(image) * 4.0
    ivar[15, 15] = 0.0
    image_hdu = fits.PrimaryHDU(image)
    image_hdu.header["CDELT1"] = -0.262 / 3600.0
    image_hdu.header["CDELT2"] = 0.262 / 3600.0
    image_hdu.header["CRPIX1"] = 16.0
    image_hdu.header["CRPIX2"] = 16.0
    image_hdu.header["CRVAL1"] = 1.2
    image_hdu.header["CRVAL2"] = 3.4
    image_hdu.header["CTYPE1"] = "RA---TAN"
    image_hdu.header["CTYPE2"] = "DEC--TAN"
    image_path = tmp_path / "legacysurvey-0000p004-image-i.fits"
    invvar_path = tmp_path / "legacysurvey-0000p004-invvar-i.fits"
    image_hdu.writeto(image_path)
    fits.PrimaryHDU(ivar).writeto(invvar_path)

    image_read, header = read_legacy_survey_coadd_image(image_path)
    band = load_legacy_survey_coadd_band(
        image_path,
        invvar_path,
        filter_name="subaru.suprime.i",
        target_ra_dec=(1.2, 3.4),
        radius=5,
        psf=np.ones((5, 5), dtype=float),
    )

    assert image_read.shape == image.shape
    assert np.isclose(abs(float(header["CDELT1"])) * 3600.0, 0.262)
    assert np.allclose(band.noise[band.mask], 0.5)
    assert int(np.size(band.mask) - np.count_nonzero(band.mask)) == 1
    assert np.isclose(band.pixel_scale, 0.262)
    assert np.isclose(band.counts_per_mjy, nanomaggy_counts_per_mjy())


def test_legacy_survey_coadd_reader_accepts_compressed_image_extension(tmp_path):
    image = np.ones((21, 21), dtype=float) * 7.0
    header = fits.Header()
    header["CDELT1"] = -0.262 / 3600.0
    header["CDELT2"] = 0.262 / 3600.0
    path = tmp_path / "legacysurvey-0000p004-image-g.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(image, header=header)]).writeto(path)

    image_read, header_read = read_legacy_survey_coadd_image(path)

    assert image_read.shape == image.shape
    assert np.allclose(image_read, image)
    assert np.isclose(abs(float(header_read["CDELT1"])) * 3600.0, 0.262)


def test_build_empirical_psfs_for_bands_prefers_common_stars(tmp_path):
    brick = "0000p004"
    shape = (101, 101)
    wcs, header = _legacy_test_wcs_header(shape)
    target_ra_dec = tuple(float(v) for v in wcs.all_pix2world([[50.0, 50.0]], 1)[0])
    ivar = np.ones(shape, dtype=float)
    g_image = np.ones(shape, dtype=float) * 2.0
    r_image = np.ones(shape, dtype=float) * 2.0
    g_image += _gaussian(shape, 30, 30, 1.2, 80.0)
    r_image += _gaussian(shape, 30, 30, 1.2, 70.0)
    g_image += _gaussian(shape, 75, 30, 1.2, 140.0)
    r_image += _gaussian(shape, 25, 75, 1.2, 130.0)
    _write_legacy_test_coadds(tmp_path, brick, "g", g_image, ivar, header)
    _write_legacy_test_coadds(tmp_path, brick, "r", r_image, ivar, header)

    result = build_empirical_psfs_for_bands(
        band_specs={"g": "subaru.suprime.g", "r": "subaru.suprime.r"},
        target_ra_dec=target_ra_dec,
        data_dir=tmp_path,
        brick=brick,
        fit_radius=10,
        config=EmpiricalPsfConfig(
            psf_size=15,
            psf_search_radius=45,
            threshold_sigma=5.0,
            max_sources=1,
            target_exclusion_radius_pix=10.0,
            max_peak_percentile=100.0,
            prefer_common_stars=True,
            psf_padding_pixels=3,
        ),
    )

    assert result.brick == brick
    assert [band.filter_name for band in result.image_bands] == ["subaru.suprime.g", "subaru.suprime.r"]
    assert len(result.common_star_groups[0]) == 2
    assert set(result.bands) == {"g", "r"}
    assert np.isclose(result.bands["g"].selected_candidates[0].x_pix + result.bands["g"].search_origin[0], 30.0, atol=1.0)
    assert np.isclose(result.bands["r"].selected_candidates[0].x_pix + result.bands["r"].search_origin[0], 30.0, atol=1.0)
    assert result.bands["g"].psf.shape == (15, 15)
    assert result.bands["g"].psf_uncertainty.shape == (15, 15)
    assert result.image_bands[0].psf_uncertainty.shape == (15, 15)
    assert np.isclose(np.sum(result.bands["g"].psf), 1.0)
    assert [band.psf_padding_pixels for band in result.image_bands] == [3, 3]


def test_build_empirical_psfs_for_galex_bands_uses_empirical_psf_pipeline(monkeypatch, tmp_path):
    target_ra_dec = (180.737219, -20.934155)
    shape = (101, 101)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [51.0, 51.0]
    wcs.wcs.crval = list(target_ra_dec)
    wcs.wcs.cdelt = [-1.5 / 3600.0, 1.5 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()

    def write_galex_image(path, amp):
        image = np.ones(shape, dtype=float) * 0.01
        image[:8, :] = 0.0
        image += _gaussian(shape, 30, 30, 1.2, amp)
        image += _gaussian(shape, 70, 30, 1.2, 0.8 * amp)
        image += _gaussian(shape, 51, 51, 1.5, 2.0 * amp)
        image[:8, :] = 0.0
        fits.PrimaryHDU(image, header=header).writeto(path, overwrite=True)

    class FakeObservations:
        @staticmethod
        def query_criteria(**_kwargs):
            return Table(rows=[("obs",)], names=("obsid",))

        @staticmethod
        def get_product_list(_observations):
            return Table(
                rows=[
                    ("FUV", "SCIENCE", "fuv-fd-int.fits.gz", "mast:GALEX/url/data/d/01-main/fuv", "1"),
                    ("NUV", "SCIENCE", "nuv-nd-int.fits.gz", "mast:GALEX/url/data/d/01-main/nuv", "1"),
                ],
                names=("filters", "productType", "productFilename", "dataURI", "obsID"),
            )

        @staticmethod
        def download_file(uri, *, local_path=None, **_kwargs):
            write_galex_image(local_path, 80.0 if uri.endswith("fuv") else 100.0)

    _install_fake_mast_observations(monkeypatch, FakeObservations)

    result = build_empirical_psfs_for_galex_bands(
        target_ra_dec=target_ra_dec,
        data_dir=tmp_path,
        fit_radius=10,
        config=EmpiricalPsfConfig(
            psf_size=15,
            psf_search_radius=45,
            threshold_sigma=5.0,
            max_sources=1,
            target_exclusion_radius_pix=10.0,
            prefer_common_stars=True,
            min_common_bands=1,
            psf_padding_pixels=3,
        ),
    )

    assert result.brick == "GALEX"
    assert set(result.bands) == {"FUV", "NUV"}
    assert [band.filter_name for band in result.image_bands] == ["galex.FUV", "galex.NUV"]
    assert all(band.image.shape == (21, 21) for band in result.image_bands)
    assert all(band.psf.shape == (15, 15) for band in result.image_bands)
    assert all(band.psf_uncertainty.shape == (15, 15) for band in result.image_bands)
    assert all(np.isclose(np.sum(band.psf), 1.0) for band in result.image_bands)
    assert all(band.counts_per_mjy > 0.0 for band in result.image_bands)
    assert result.bands["FUV"].invvar_path is None
    assert result.bands["FUV"].search_mask.shape == result.bands["FUV"].search_image.shape
    assert not np.all(result.bands["FUV"].search_mask)


def test_combine_empirical_psf_results_keeps_all_bands(tmp_path):
    psf = np.ones((5, 5), dtype=float) / 25.0
    band_a = ImageBandData(np.ones((7, 7)), np.ones((7, 7)), psf, "a", pixel_scale=1.0)
    band_b = ImageBandData(np.ones((7, 7)), np.ones((7, 7)), psf, "b", pixel_scale=1.0)
    candidate = PsfCandidate(3.0, 3.0, 10.0, 1.0, 10.0)
    wcs = WCS(naxis=2)
    result_a = EmpiricalPsfResult(
        brick="A",
        image_bands=[band_a],
        bands={
            "a": EmpiricalPsfBandResult(
                band_code="a",
                filter_name="a",
                image_path=tmp_path / "a.fits",
                invvar_path=None,
                psf=psf,
                candidates=[candidate],
                selected_candidates=[candidate],
                search_image=np.ones((7, 7)),
                search_target_pixel=(3.0, 3.0),
                search_wcs=wcs,
                full_target_pixel=(3.0, 3.0),
                search_origin=(0, 0),
            )
        },
        common_star_groups=[{"a": candidate}],
        config=EmpiricalPsfConfig(),
    )
    result_b = EmpiricalPsfResult(
        brick="B",
        image_bands=[band_b],
        bands={
            "b": EmpiricalPsfBandResult(
                band_code="b",
                filter_name="b",
                image_path=tmp_path / "b.fits",
                invvar_path=None,
                psf=psf,
                candidates=[candidate],
                selected_candidates=[candidate],
                search_image=np.ones((7, 7)),
                search_target_pixel=(3.0, 3.0),
                search_wcs=wcs,
                full_target_pixel=(3.0, 3.0),
                search_origin=(0, 0),
            )
        },
        common_star_groups=[{"b": candidate}],
        config=EmpiricalPsfConfig(),
    )

    combined = combine_empirical_psf_results(result_a, result_b)

    assert combined.brick == "A+B"
    assert [band.filter_name for band in combined.image_bands] == ["a", "b"]
    assert set(combined.bands) == {"a", "b"}
    assert len(combined.common_star_groups) == 2
