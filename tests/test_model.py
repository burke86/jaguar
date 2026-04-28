from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from jaguar import ComponentFluxes, ImageBandData, ImageFitConfig, JointFitConfig, SceneComponentConfig, SedComponentConfig, build_grahspj_config_from_image_bands
from jaguar.model import _blackbody_state, _combine_component_states, _component_grahspj_config, render_joint_model


def _band(mask=None):
    image = np.zeros((21, 21), dtype=float)
    noise = np.ones_like(image)
    psf = np.ones((5, 5), dtype=float)
    return ImageBandData(image=image, noise=noise, psf=psf, filter_name="hsc_i", pixel_scale=0.168, mask=mask)


def test_band_model_is_component_sum():
    fluxes = {"hsc_i": ComponentFluxes(agn=10.0, host=20.0)}
    cfg = JointFitConfig(
        image_bands=[_band()],
        image=ImageFitConfig(fit_background=False),
        grahspj_config=object(),
    )
    rendered = render_joint_model(cfg, {}, fluxes_by_band=fluxes)
    comp = rendered["hsc_i"]
    assert np.allclose(comp["total"], comp["agn"] + comp["star"] + comp["host"] + comp["background"])
    assert np.isclose(np.sum(comp["agn"]), 10.0)
    assert np.isclose(np.sum(comp["host"]), 20.0)


def test_config_rejects_missing_grahspj_config():
    cfg = JointFitConfig(image_bands=[_band()])
    try:
        cfg.validate()
    except ValueError as exc:
        assert "grahspj_config or sed_components is required" in str(exc)
    else:
        raise AssertionError("Expected missing flux validation error.")


def test_default_sed_components_split_agn_and_host_configs():
    band = ImageBandData(
        np.ones((5, 5), dtype=float),
        np.ones((5, 5), dtype=float),
        np.ones((3, 3), dtype=float),
        "hsc_i",
        pixel_scale=0.168,
        counts_per_mjy=1.0,
    )
    grahspj_config = build_grahspj_config_from_image_bands([band], dsps_ssp_fn="/tmp/not-used.h5")
    cfg = JointFitConfig(image_bands=[band], grahspj_config=grahspj_config)

    components = cfg.resolved_sed_components
    assert [(component.name, component.kind, component.resolved_spatial) for component in components] == [
        ("agn", "agn", "point"),
        ("host", "host", "extended"),
    ]

    agn_config = _component_grahspj_config(components[0])
    host_config = _component_grahspj_config(components[1])
    assert agn_config.agn.fit_agn
    assert not agn_config.galaxy.fit_host
    assert not agn_config.nebular.enabled
    assert host_config.galaxy.fit_host
    assert not host_config.agn.fit_agn


def test_star_component_defaults_to_point_source():
    component = SedComponentConfig(name="star_1", kind="star")
    component.validate()
    assert component.resolved_spatial == "point"


def test_sersic_component_rejects_invalid_reff_bounds():
    component = SceneComponentConfig(
        name="host",
        sed_component="host",
        kind="sersic",
        min_reff_arcsec=0.5,
        max_reff_arcsec=0.4,
    )
    try:
        component.validate()
    except ValueError as exc:
        assert "max_reff_arcsec" in str(exc)
    else:
        raise AssertionError("Expected invalid reff bounds validation error.")


def test_arbitrary_scene_components_render_independently():
    band = _band()
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_background=False),
        sed_components=[
            SedComponentConfig(name="star_1", kind="star"),
            SedComponentConfig(name="star_2", kind="star"),
        ],
        scene_components=[
            SceneComponentConfig(name="star_1_image", sed_component="star_1", kind="point", fit_position=False, fixed_center_x_pix=-4.0, fixed_center_y_pix=-4.0),
            SceneComponentConfig(name="star_2_image", sed_component="star_2", kind="point", fit_position=False, fixed_center_x_pix=4.0, fixed_center_y_pix=4.0),
        ],
    )
    rendered = render_joint_model(
        cfg,
        {},
        fluxes_by_band={"hsc_i": {"star_1_image": 5.0, "star_2_image": 7.0}},
    )["hsc_i"]

    assert np.isclose(np.sum(rendered["star_1_image"]), 5.0)
    assert np.isclose(np.sum(rendered["star_2_image"]), 7.0)
    assert np.isclose(np.sum(rendered["star"]), 12.0)
    assert np.allclose(rendered["total"], rendered["star"] + rendered["background"])


def test_point_source_psf_uncertainty_renders_flux_scaled_variance():
    psf_uncertainty = np.ones((5, 5), dtype=float) * 0.01
    band = ImageBandData(
        image=np.zeros((21, 21), dtype=float),
        noise=np.ones((21, 21), dtype=float),
        psf=np.ones((5, 5), dtype=float),
        psf_uncertainty=psf_uncertainty,
        filter_name="hsc_i",
        pixel_scale=0.168,
    )
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_background=False),
        sed_components=[SedComponentConfig(name="agn", kind="agn", grahspj_config=object())],
        scene_components=[SceneComponentConfig(name="agn_image", sed_component="agn", kind="point", fit_position=False)],
    )

    rendered = render_joint_model(cfg, {}, fluxes_by_band={"hsc_i": {"agn_image": 10.0}})["hsc_i"]

    assert np.max(rendered["psf_variance"]) > 0.0
    assert np.isclose(np.max(rendered["psf_variance"]), 0.01)


def test_mixed_grahspj_and_star_components_use_full_sed_grid_for_plotting():
    band = ImageBandData(
        np.zeros((11, 11), dtype=float),
        np.ones((11, 11), dtype=float),
        np.ones((3, 3), dtype=float),
        "hsc_i",
        pixel_scale=0.168,
    )
    cfg = JointFitConfig(
        image_bands=[band],
        sed_components=[
            SedComponentConfig(name="agn", kind="agn", grahspj_config=object()),
            SedComponentConfig(name="star", kind="star", reference_flux_mjy=2.0, fit_reference_flux=False),
        ],
        scene_components=[
            SceneComponentConfig(name="agn_image", sed_component="agn", kind="point"),
            SceneComponentConfig(name="star_image", sed_component="star", kind="point"),
        ],
    )
    cfg.grahspj_context = SimpleNamespace(
        rest_wave_jax=np.geomspace(100.0, 1.0e5, 512),
        obs_wave_jax=np.geomspace(150.0, 1.5e5, 512),
        filter_effective_wavelength_jax=np.asarray([7693.0]),
    )
    cfg.resolved_sed_components[0].grahspj_context = cfg.grahspj_context
    agn_state = {
        "pred_fluxes": np.asarray([1.0]),
        "total_rest_sed": np.ones(512),
        "total_obs_sed": np.ones(512),
    }
    star_state = _blackbody_state(cfg, cfg.resolved_sed_components[1])
    combined = _combine_component_states(cfg, {"agn": agn_state, "star": star_state}, include_components=True)

    assert combined["total_obs_sed"].shape == (512,)
    assert combined["total_rest_sed"].shape == (512,)
    assert np.allclose(np.asarray(combined["component_fluxes"]["star"]), [2.0])


def test_sersic_scene_component_is_psf_convolved_and_flux_preserved():
    image = np.zeros((41, 41), dtype=float)
    noise = np.ones_like(image)
    psf_delta = np.zeros((9, 9), dtype=float)
    psf_delta[4, 4] = 1.0
    psf_smooth = np.ones((9, 9), dtype=float)
    band_delta = ImageBandData(image=image, noise=noise, psf=psf_delta, filter_name="hsc_i", pixel_scale=0.168)
    band_smooth = ImageBandData(image=image, noise=noise, psf=psf_smooth, filter_name="hsc_i", pixel_scale=0.168)
    cfg_delta = JointFitConfig(
        image_bands=[band_delta],
        image=ImageFitConfig(fit_background=False),
        sed_components=[SedComponentConfig(name="host", kind="host", grahspj_config=object())],
        scene_components=[
            SceneComponentConfig(name="host_image", sed_component="host", kind="sersic", fit_position=False, fit_shape=False, fixed_reff_arcsec=0.2),
        ],
    )
    cfg_smooth = JointFitConfig(
        image_bands=[band_smooth],
        image=ImageFitConfig(fit_background=False),
        sed_components=[SedComponentConfig(name="host", kind="host", grahspj_config=object())],
        scene_components=[
            SceneComponentConfig(name="host_image", sed_component="host", kind="sersic", fit_position=False, fit_shape=False, fixed_reff_arcsec=0.2),
        ],
    )

    delta = render_joint_model(cfg_delta, {}, fluxes_by_band={"hsc_i": {"host_image": 10.0}})["hsc_i"]["host_image"]
    smooth = render_joint_model(cfg_smooth, {}, fluxes_by_band={"hsc_i": {"host_image": 10.0}})["hsc_i"]["host_image"]

    assert np.isclose(np.sum(delta), 10.0)
    assert np.isclose(np.sum(smooth), 10.0)
    assert float(np.max(smooth)) < float(np.max(delta))


def test_masked_pixels_do_not_change_rendered_model():
    mask = np.ones((21, 21), dtype=bool)
    mask[0, 0] = False
    cfg = JointFitConfig(
        image_bands=[_band(mask=mask)],
        image=ImageFitConfig(fit_background=False),
        grahspj_config=object(),
    )
    rendered = render_joint_model(cfg, {}, fluxes_by_band={"hsc_i": ComponentFluxes(agn=10.0, host=20.0)})["hsc_i"]["total"]
    assert rendered.shape == (21, 21)
