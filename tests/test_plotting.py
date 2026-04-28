from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace

from jaguar import ComponentFluxes, DetectedSource, ImageBandData, ImageFitConfig, JointFitConfig, SceneComponentConfig, SedComponentConfig, SourceDetectionConfig
from jaguar.io import EmpiricalPsfBandResult, EmpiricalPsfConfig, EmpiricalPsfResult, PsfCandidate
from jaguar.model import render_joint_model
from jaguar.plotting import _radial_surface_brightness_profile, plot_config, plot_empirical_psf_selection, plot_fit, plot_psf_candidates, plot_sed
from jaguar.result import JaguarResult


def _result():
    shape = (21, 21)
    psf = np.ones((5, 5), dtype=float)
    band = ImageBandData(np.ones(shape), np.ones(shape), psf, "hsc_i", pixel_scale=0.168, zeropoint=22.5)
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


def _multi_host_result():
    shape = (31, 31)
    psf = np.ones((5, 5), dtype=float)
    band = ImageBandData(np.ones(shape), np.ones(shape), psf, "hsc_i", pixel_scale=0.168, zeropoint=22.5)
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_background=False),
        sed_components=[
            SedComponentConfig(name="agn", kind="agn", grahspj_config=object()),
            SedComponentConfig(name="host_inner", kind="host", grahspj_config=object()),
            SedComponentConfig(name="host_outer", kind="host", grahspj_config=object()),
            SedComponentConfig(name="det_1", kind="galaxy", grahspj_config=object()),
        ],
        scene_components=[
            SceneComponentConfig(name="agn", sed_component="agn", kind="point", fit_position=False),
            SceneComponentConfig(name="host_inner", sed_component="host_inner", kind="sersic", fit_position=False, fit_shape=False, fixed_reff_arcsec=0.3),
            SceneComponentConfig(name="host_outer", sed_component="host_outer", kind="sersic", fit_position=False, fit_shape=False, fixed_reff_arcsec=1.0),
            SceneComponentConfig(name="det_1_image", sed_component="det_1", kind="sersic", fit_position=False, fit_shape=False, fixed_center_x_pix=4.0, fixed_reff_arcsec=0.5),
        ],
    )
    rendered = render_joint_model(
        cfg,
        {},
        fluxes_by_band={"hsc_i": {"agn": 10.0, "host_inner": 5.0, "host_outer": 15.0, "det_1_image": 8.0}},
    )
    state = {
        "pred_fluxes": np.asarray([30.0]),
        "agn_fluxes": np.asarray([10.0]),
        "host_fluxes": np.asarray([20.0]),
        "dust_fluxes": np.asarray([0.0]),
        "nebular_fluxes": np.asarray([0.0]),
    }
    return JaguarResult(config=cfg, map_params={}, samples=None, rendered=rendered, grahspj_state=state)


def test_plot_fit_accepts_shared_log_limits():
    result = _result()
    fig, axes = plot_fit(result, shared_vmin=1.0e-3, shared_vmax=10.0)
    norms = [ax.images[0].norm for ax in axes[:3]]
    assert all(norm.vmin == 1.0e-3 for norm in norms)
    assert all(norm.vmax == 10.0 for norm in norms)
    labels = [ax.texts[0].get_text() for ax in axes[:4]]
    assert labels == ["Data", "Model", "Data - Point Source", "Residual"]
    assert axes[-2].get_ylabel() == r"$\mu$ (mag arcsec$^{-2}$)"
    assert axes[-2].yaxis.get_label_position() == "right"
    assert axes[-2].yaxis.label.get_size() == 10
    assert axes[-1].get_xlabel() == "Radius (arcsec)"
    assert axes[-1].get_ylabel() == r"$\Delta\mu$"
    assert axes[-1].yaxis.get_label_position() == "right"
    assert axes[-1].yaxis.label.get_size() == 10
    assert axes[-2].yaxis_inverted()
    assert axes[-2].collections
    assert axes[-1].collections
    assert not any(line.get_marker() == "o" and line.get_linestyle() == "-" for line in axes[-1].lines)
    assert not any(line.get_visible() for line in axes[-2].get_xgridlines())
    assert not any(line.get_visible() for line in axes[-1].get_xgridlines())
    assert len(fig.axes) == len(axes) + 2
    assert fig.axes[-2].get_ylabel() == "counts"
    assert fig.axes[-1].get_ylabel() == r"$(data-model)/\sigma$"
    fig.clear()


def test_plot_fit_surface_brightness_shows_each_host_component():
    result = _multi_host_result()

    fig, axes = plot_fit(result)

    labels = [line.get_label() for line in axes[-2].lines]
    assert "Host: host_inner" in labels
    assert "Host: host_outer" in labels
    assert "Host: det_1_image" not in labels
    assert "Host" not in labels
    fig.clear()


def test_plot_fit_warns_when_surface_brightness_zeropoint_is_assumed():
    result = _result()
    result.config.image_bands = [replace(result.config.image_bands[0], zeropoint=None)]

    with pytest.warns(RuntimeWarning, match="assuming AB zeropoint 22.5"):
        fig, axes = plot_fit(result)

    assert any(text.get_text() == "Assumed ZP=22.5" for text in axes[-2].texts)
    fig.clear()


def test_plot_fit_profile_radius_clips_to_host_size():
    result = _result()
    fig, axes = plot_fit(result, profile_radius_factor=2.0)

    assert axes[-1].get_xlim()[1] <= 1.05
    fig.clear()


def test_plot_fit_profile_and_residual_axes_touch():
    result = _result()
    fig, axes = plot_fit(result)
    fig.canvas.draw()

    image_box = axes[0].get_position()
    profile_box = axes[-2].get_position()
    residual_box = axes[-1].get_position()
    assert abs(profile_box.y0 - residual_box.y1) < 1.0e-12
    assert abs(profile_box.y1 - image_box.y1) < 1.0e-12
    assert abs(residual_box.y0 - image_box.y0) < 1.0e-12
    assert abs(profile_box.width - image_box.width) < 1.0e-12
    fig.clear()


def test_surface_brightness_profile_masks_non_positive_mag_bins():
    image = np.ones((9, 9), dtype=float)
    image[:, 5:] = 0.0

    _radius, profile, error = _radial_surface_brightness_profile(
        image,
        1.0,
        zeropoint=22.5,
        center_xy=(0.0, 4.0),
        bin_width_arcsec=1.0,
    )

    assert np.nanmax(profile) < 100.0
    assert np.any(~np.isfinite(profile))
    assert np.all(~np.isfinite(error[~np.isfinite(profile)]))


def test_plot_fit_surface_brightness_model_excludes_background():
    result = _result()
    band = result.config.image_bands[0]
    result.rendered[band.filter_name]["total"] = result.rendered[band.filter_name]["total"] - 100.0
    result.rendered[band.filter_name]["background"] = np.ones_like(band.image) * -100.0

    fig, axes = plot_fit(result)

    model_line = next(line for line in axes[-2].lines if line.get_label() == "Model")
    assert np.all(np.isfinite(model_line.get_ydata()))
    assert np.nanmax(model_line.get_ydata()) < 100.0
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


def test_plot_config_draws_one_row_per_band_with_target_images_and_psfs():
    shape = (21, 21)
    band_i = ImageBandData(
        np.ones(shape, dtype=float) * 2.0,
        np.ones(shape, dtype=float),
        np.array([[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]], dtype=float),
        "hsc_i",
        pixel_scale=0.168,
    )
    band_g = ImageBandData(
        np.ones(shape, dtype=float),
        np.ones(shape, dtype=float),
        np.array([[0.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 0.0]], dtype=float),
        "hsc_g",
        pixel_scale=0.168,
    )
    cfg = SourceDetectionConfig(target_exclusion_radius_pix=4.0)
    detections = [
        DetectedSource(1, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 5, 1.0, 1.0, 1.0, "star"),
        DetectedSource(2, 16.0, 10.0, 6.0, 0.0, 10.0, 10.0, 5, 2.0, 1.0, 2.0, "galaxy"),
    ]

    fig, axes = plot_config([band_i, band_g], detections, cfg, detection_image=np.ones(shape, dtype=float))
    ax = axes[0, 0]

    assert axes.shape == (2, 3)
    assert len(ax.patches) == 1
    assert len(ax.lines) == 1
    assert ax.images[0].cmap.name == "viridis"
    assert axes[0, 0].get_title() == "Target image: hsc_i"
    assert axes[1, 0].get_title() == "Target image: hsc_g"
    assert axes[0, 1].get_title() == "PSF: hsc_i"
    assert axes[1, 1].get_title() == "PSF: hsc_g"
    assert axes[0, 1].images[0].cmap.name == "viridis"
    assert axes[0, 2].get_title() == "PSF profile: hsc_i"
    assert axes[1, 2].get_title() == "PSF profile: hsc_g"
    assert axes[0, 2].get_xlabel() == "Radius (arcsec)"
    assert axes[0, 2].get_ylabel() == r"$\mu_{\rm rel}$ (mag arcsec$^{-2}$)"
    assert axes[0, 2].yaxis.get_label_position() == "right"
    assert axes[0, 2].yaxis.label.get_size() == 10
    assert axes[0, 2].yaxis_inverted()
    assert np.all(np.isfinite(axes[0, 2].lines[0].get_ydata()))
    fig.clear()


def test_plot_psf_candidates_circles_candidates_and_target_exclusion():
    image = np.ones((41, 41), dtype=float)
    candidates = [PsfCandidate(x_pix=12.0, y_pix=14.0, flux=10.0, size_pix=1.0, peak=5.0)]

    fig, ax = plot_psf_candidates(
        image,
        candidates,
        target_pixel=(20.0, 20.0),
        psf_size=11,
        target_exclusion_radius_pix=5.0,
    )

    assert len(ax.patches) == 3
    assert ax.images[0].cmap.name == "viridis"
    fig.clear()


def test_plot_psf_candidates_marks_gaia_stars_red_with_legend():
    image = np.ones((41, 41), dtype=float)
    candidates = [PsfCandidate(x_pix=12.0, y_pix=14.0, flux=10.0, size_pix=1.0, peak=5.0, is_gaia_star=True)]

    fig, ax = plot_psf_candidates(
        image,
        candidates,
        target_pixel=(20.0, 20.0),
        psf_size=11,
        target_exclusion_radius_pix=5.0,
    )

    assert ax.patches[1].get_edgecolor() == (1.0, 0.0, 0.0, 1.0)
    assert ax.get_legend().texts[0].get_text() == "Gaia star"
    fig.clear()


def test_plot_empirical_psf_selection_uses_common_result():
    image = np.ones((41, 41), dtype=float)
    selected = PsfCandidate(x_pix=12.0, y_pix=14.0, flux=10.0, size_pix=1.0, peak=5.0)
    other = PsfCandidate(x_pix=25.0, y_pix=24.0, flux=8.0, size_pix=1.0, peak=4.0)
    band = EmpiricalPsfBandResult(
        band_code="g",
        filter_name="subaru.suprime.g",
        image_path=SimpleNamespace(),
        invvar_path=SimpleNamespace(),
        psf=np.ones((5, 5), dtype=float) / 25.0,
        psf_uncertainty=np.zeros((5, 5), dtype=float),
        candidates=[selected, other],
        selected_candidates=[selected],
        search_image=image,
        search_target_pixel=(20.0, 20.0),
        search_wcs=SimpleNamespace(),
        full_target_pixel=(20.0, 20.0),
        search_origin=(0, 0),
    )
    result = EmpiricalPsfResult(
        brick="0000p004",
        image_bands=[],
        bands={"g": band},
        common_star_groups=[{"g": selected}],
        config=EmpiricalPsfConfig(psf_size=11, target_exclusion_radius_pix=5.0),
    )

    fig, axes = plot_empirical_psf_selection(result, show_all_candidates=True)

    assert len(axes) == 1
    assert axes[0].get_title() == "PSF candidates: g"
    assert axes[0].get_legend().texts[0].get_text() == "Selected PSF star"
    assert len(axes[0].patches) == 4
    fig.clear()
