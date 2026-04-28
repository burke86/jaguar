from __future__ import annotations

import numpy as np
import importlib
import jax
import numpyro.distributions as dist
from types import SimpleNamespace

from jaguar import ComponentFluxes, ImageBandData, ImageFitConfig, JointFitConfig, build_grahspj_config_from_image_bands, fit
from jaguar.fit import _init_to_values_then_median, _initial_values
from jaguar.model import render_joint_model


def _fake_grahspj(monkeypatch, agn=10.0, host=20.0):
    fit_mod = importlib.import_module("jaguar.fit")
    import jaguar.model as model_mod

    def ensure(config):
        config.grahspj_context = SimpleNamespace(
            fit_config=SimpleNamespace(photometry=SimpleNamespace(filter_names=["hsc_i"]))
        )

    def state(config, sampled=None, *, add_likelihood=True, include_components=False):
        return {
            "pred_fluxes": np.asarray([agn + host]),
            "agn_fluxes": np.asarray([agn]),
            "host_fluxes": np.asarray([host]),
            "star_fluxes": np.asarray([0.0]),
            "component_fluxes": {"agn": np.asarray([agn]), "host": np.asarray([host])},
            "dust_fluxes": np.asarray([0.0]),
            "nebular_fluxes": np.asarray([0.0]),
            "rest_wave": np.asarray([1.0]),
            "obs_wave": np.asarray([1.0]),
            "total_rest_sed": np.asarray([1.0]),
            "agn_rest_sed": np.asarray([1.0]),
            "host_rest_sed": np.asarray([1.0]),
            "host_total_rest_sed": np.asarray([1.0]),
            "dust_rest_sed": np.asarray([0.0]),
            "nebular_rest_sed": np.asarray([0.0]),
            "total_obs_sed": np.asarray([1.0]),
        }

    monkeypatch.setattr(model_mod, "ensure_grahspj_context", ensure)
    monkeypatch.setattr(fit_mod, "ensure_grahspj_context", ensure)
    monkeypatch.setattr(model_mod, "_grahspj_state_from_trace", state)
    monkeypatch.setattr(fit_mod, "_grahspj_state_from_trace", state)


def test_map_fit_smoke(monkeypatch):
    _fake_grahspj(monkeypatch)
    shape = (21, 21)
    psf = np.ones((5, 5), dtype=float)
    noise = np.ones(shape, dtype=float) * 5.0
    blank = np.zeros(shape, dtype=float)
    band = ImageBandData(blank, noise, psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=1.0)
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_host_morphology=False, fit_background=False),
        grahspj_config=object(),
    )
    image = np.asarray(render_joint_model(cfg, {}, fluxes_by_band={"hsc_i": ComponentFluxes(agn=10.0, host=20.0)})["hsc_i"]["total"])
    band = ImageBandData(image, noise, psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=1.0)
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_host_morphology=False, fit_background=False),
        grahspj_config=object(),
    )
    result = fit(cfg, fit_method="map_only", map_steps=2, progress_bar=False)
    summary = result.summary()
    assert np.isclose(summary["hsc_i_agn_flux"], 10.0)
    assert np.isclose(summary["hsc_i_host_flux"], 20.0)


def test_map_fit_accepts_integer_background_default(monkeypatch):
    _fake_grahspj(monkeypatch, agn=1.0, host=2.0)
    shape = (11, 11)
    psf = np.ones((5, 5), dtype=float)
    noise = np.ones(shape, dtype=float)
    band = ImageBandData(np.zeros(shape), noise, psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=1.0)
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_host_morphology=False, fit_background=True, background_default=0),
        grahspj_config=object(),
    )
    result = fit(cfg, fit_method="map_only", map_steps=1, progress_bar=False)
    assert "background_0" in result.map_params


def test_map_fit_with_free_morphology_initializes(monkeypatch):
    _fake_grahspj(monkeypatch, agn=1.0, host=2.0)
    shape = (21, 21)
    psf = np.ones((5, 5), dtype=float)
    noise = np.ones(shape, dtype=float) * 5.0
    band = ImageBandData(np.zeros(shape), noise, psf, "hsc_i", pixel_scale=0.168, counts_per_mjy=1.0)
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_host_morphology=True, fit_background=True, background_default=0),
        grahspj_config=object(),
    )
    result = fit(cfg, fit_method="map_only", map_steps=1, progress_bar=False)
    assert "host/reff_arcsec" in result.map_params


def test_grahspj_config_does_not_override_log_agn_amp_with_log10_scale():
    shape = (11, 11)
    band = ImageBandData(
        image=np.ones(shape, dtype=float),
        noise=np.ones(shape, dtype=float),
        psf=np.ones((5, 5), dtype=float),
        filter_name="subaru.suprime.i",
        pixel_scale=0.168,
        counts_per_mjy=100.0,
    )
    cfg = build_grahspj_config_from_image_bands([band], dsps_ssp_fn="/tmp/not-used.h5")
    assert "log_agn_amp" not in cfg.prior_config


def test_unspecified_grahspj_sites_initialize_at_prior_median():
    shape = (11, 11)
    band = ImageBandData(
        image=np.zeros(shape, dtype=float),
        noise=np.ones(shape, dtype=float),
        psf=np.ones((5, 5), dtype=float),
        filter_name="hsc_i",
        pixel_scale=0.168,
    )
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_host_morphology=True, fixed_reff_arcsec=0.45),
        grahspj_config=object(),
    )
    init = _init_to_values_then_median(_initial_values(cfg))
    assert float(init({"type": "sample", "is_observed": False, "name": "host/reff_arcsec"})) == 0.45
    value = init(
        {
            "type": "sample",
            "is_observed": False,
            "name": "log_stellar_mass",
            "fn": dist.Normal(9.5, 1.5),
            "value": None,
            "kwargs": {"rng_key": jax.random.PRNGKey(0), "sample_shape": ()},
        }
    )
    assert 7.0 < float(value) < 12.0
