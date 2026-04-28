from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from jaguar import ComponentFluxes, DetectedSource, ImageBandData, ImageFitConfig, JointFitConfig, SourceDetectionConfig
from jaguar.model import render_joint_model
from jaguar.plotting import plot_detection, plot_fit, plot_sed
from jaguar.result import JaguarResult


def _result():
    shape = (21, 21)
    psf = np.ones((5, 5), dtype=float)
    band = ImageBandData(np.ones(shape), np.ones(shape), psf, "hsc_i", pixel_scale=0.168)
    grahspj_config = SimpleNamespace(
        photometry=SimpleNamespace(
            fluxes=[30.0],
            errors=[3.0],
            filter_names=["hsc_i"],
        ),
        likelihood=SimpleNamespace(
            systematics_width=0.0,
            variability_uncertainty=False,
            attenuation_model_uncertainty=False,
            lyman_break_uncertainty=False,
        ),
    )
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_background=False),
        grahspj_config=grahspj_config,
    )
    cfg.grahspj_context = SimpleNamespace(
        filters=[SimpleNamespace(effective_wavelength=7693.0)],
        filter_effective_wavelength_jax=np.asarray([7693.0]),
        errors=np.asarray([3.0]),
        upper_limits=np.asarray([False]),
    )
    rendered = render_joint_model(cfg, {}, fluxes_by_band={"hsc_i": ComponentFluxes(agn=10.0, host=20.0)})
    state = {
        "pred_fluxes": np.asarray([30.0]),
        "agn_fluxes": np.asarray([10.0]),
        "host_fluxes": np.asarray([20.0]),
        "dust_fluxes": np.asarray([0.0]),
        "nebular_fluxes": np.asarray([0.0]),
        "obs_wave": np.linspace(1.0e3, 1.0e5, 32),
        "redshift_fit": np.asarray(0.5),
        "total_obs_sed": np.ones(32) * 1.0e-20,
        "agn_obs_sed": np.ones(32) * 3.0e-21,
        "host_obs_sed": np.ones(32) * 7.0e-21,
        "component_states": {
            "agn": {
                "kind": "agn",
                "pred_fluxes": np.asarray([10.0]),
                "total_obs_sed": np.ones(32) * 3.0e-21,
            },
            "host": {
                "kind": "host",
                "pred_fluxes": np.asarray([20.0]),
                "total_obs_sed": np.ones(32) * 7.0e-21,
            },
        },
    }
    return JaguarResult(config=cfg, map_params={}, samples=None, rendered=rendered, grahspj_state=state)


def test_plot_fit_accepts_shared_log_limits():
    result = _result()
    fig, axes = plot_fit(result, shared_vmin=1.0e-3, shared_vmax=10.0)
    norms = [ax.images[0].norm for ax in axes[:4]]
    assert all(norm.vmin == 1.0e-3 for norm in norms)
    assert all(norm.vmax == 10.0 for norm in norms)
    fig.clear()


def test_plot_sed_calls_grahspj_sed_plotter():
    result = _result()
    fig = plot_sed(result)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].get_yscale() == "log"
    assert len(result.sed_points()) == 1
    fig.clear()


def test_plot_sed_can_select_named_component():
    result = _result()
    fig = plot_sed(result, component="agn")
    assert len(fig.axes) == 2
    fig.clear()


def test_plot_sed_rejects_unknown_component():
    result = _result()
    try:
        plot_sed(result, component="missing")
    except KeyError as exc:
        assert "Available components" in str(exc)
    else:
        raise AssertionError("Expected unknown SED component to raise KeyError.")


def test_plot_sed_prefers_grahspj_filter_wavelengths():
    result = _result()
    result.config.grahspj_context.filter_effective_wavelength_jax = np.asarray([7693.0])
    assert result.sed_points()[0].wavelength == 7693.0


def test_plot_detection_draws_exclusion_radius_and_skips_central_sources():
    shape = (21, 21)
    band = ImageBandData(
        np.ones(shape, dtype=float),
        np.ones(shape, dtype=float),
        np.ones((3, 3), dtype=float),
        "hsc_i",
        pixel_scale=0.168,
    )
    cfg = SourceDetectionConfig(target_exclusion_radius_pix=4.0)
    detections = [
        DetectedSource(1, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 5, 1.0, 1.0, 1.0, "star"),
        DetectedSource(2, 16.0, 10.0, 6.0, 0.0, 10.0, 10.0, 5, 2.0, 1.0, 2.0, "galaxy"),
    ]

    fig, ax = plot_detection([band], detections, cfg, detection_image=np.ones(shape, dtype=float))

    assert len(ax.patches) == 1
    assert len(ax.lines) == 1
    assert ax.images[0].cmap.name == "viridis"
    fig.clear()
