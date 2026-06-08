from __future__ import annotations

import os

import pytest

from jaguar.io import download_galex_coadd_blocks, query_galex_coadd_products, read_legacy_survey_coadd_image


pytestmark = pytest.mark.skipif(
    os.environ.get("JAGUAR_RUN_LIVE_MAST") != "1",
    reason="Set JAGUAR_RUN_LIVE_MAST=1 to run the live MAST/GALEX download test.",
)


def test_download_galex_pox52_fuv_and_nuv_coadd_blocks(tmp_path):
    target_ra_dec = (180.737219, -20.934155)

    products = query_galex_coadd_products(target_ra_dec)
    assert products["FUV"]
    assert products["NUV"]

    paths = download_galex_coadd_blocks(tmp_path, target_ra_dec)

    assert set(paths) == {"FUV", "NUV"}
    for band, path in paths.items():
        assert path.exists(), band
        image, header = read_legacy_survey_coadd_image(path)
        assert image.ndim == 2
        assert image.size > 0
        assert "CTYPE1" in header
