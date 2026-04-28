from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from .config import ComponentFluxes, JointFitConfig, SceneComponentConfig, SedComponentConfig, coerce_component_fluxes
from .render import convolve_fft_same, psf_unit_flux, sersic_ellipse_unit_flux


_H_PLANCK = 6.62607015e-34
_K_BOLTZMANN = 1.380649e-23
_C_M_PER_S = 2.99792458e8
_GALAXY_KINDS = {"host", "galaxy"}


def _fixed_fluxes(config: JointFitConfig) -> dict[str, ComponentFluxes]:
    assert config.fixed_component_fluxes is not None
    return {
        name: coerce_component_fluxes(value)
        for name, value in config.fixed_component_fluxes.items()
    }


def resolve_component_fluxes(config: JointFitConfig, sampled: Mapping[str, Any]) -> Mapping[str, ComponentFluxes]:
    """Resolve per-band component fluxes from fixed config or a grahspj bridge."""

    filter_names = [band.filter_name for band in config.image_bands]
    if config.component_flux_model is not None:
        return config.component_flux_model(sampled, filter_names)
    return _fixed_fluxes(config)


def ensure_grahspj_context(config: JointFitConfig) -> None:
    """Build and cache grahspj static contexts on the mutable joint config."""

    from grahspj.preload import build_model_context

    primary_context = config.grahspj_context
    for component in config.resolved_sed_components:
        if component.kind != "agn" and component.kind not in _GALAXY_KINDS:
            continue
        if component.grahspj_context is None:
            component.grahspj_context = build_model_context(_component_grahspj_config(component))
        if primary_context is None:
            primary_context = component.grahspj_context
    if primary_context is None:
        raise ValueError("At least one grahspj SED component is required for the joint photometry likelihood.")
    config.grahspj_context = primary_context


def _component_grahspj_config(component: SedComponentConfig):
    """Return a grahspj config with component-specific host/AGN switches."""

    cfg = component.grahspj_config
    if component.kind == "agn":
        return replace(
            cfg,
            galaxy=replace(cfg.galaxy, fit_host=False, use_energy_balance=False),
            nebular=replace(cfg.nebular, enabled=False),
            agn=replace(cfg.agn, fit_agn=True),
        )
    if component.kind in _GALAXY_KINDS:
        return replace(
            cfg,
            galaxy=replace(cfg.galaxy, fit_host=True),
            agn=replace(cfg.agn, fit_agn=False),
        )
    return cfg


def _grahspj_state_from_trace(
    config: JointFitConfig,
    sampled: Mapping[str, Any] | None = None,
    *,
    add_likelihood: bool = True,
    include_components: bool = False,
) -> Mapping[str, Any]:
    """Evaluate the aggregate component SED state inside a NumPyro trace."""

    ensure_grahspj_context(config)
    component_states = _component_states_from_trace(
        config,
        sampled,
        include_components=include_components,
    )
    state = _combine_component_states(config, component_states, include_components=include_components)
    if add_likelihood:
        _add_joint_photometry_likelihood(config, state)
    return state


def _scoped_component_state(
    component: SedComponentConfig,
    *,
    include_components: bool,
) -> Mapping[str, Any]:
    """Evaluate one grahspj component under a scoped NumPyro namespace."""

    from grahspj.model import evaluate_photometric_state

    def _eval():
        return evaluate_photometric_state(
            component.grahspj_context,
            include_components=include_components,
            include_sed_agn_features=True,
            include_spectral_features=True,
            add_likelihood=False,
            force_component_fluxes=True,
        )

    return handlers.scope(_eval, prefix=component.name)()


def _filter_wavelengths(config: JointFitConfig) -> jnp.ndarray:
    """Return image filter effective wavelengths in Angstrom."""

    ensure_grahspj_context(config)
    if config.grahspj_context is not None:
        return jnp.asarray(config.grahspj_context.filter_effective_wavelength_jax, dtype=jnp.float64)
    values = [
        float(band.effective_wavelength) if band.effective_wavelength is not None else float(i + 1)
        for i, band in enumerate(config.image_bands)
    ]
    return jnp.asarray(values, dtype=jnp.float64)


def _blackbody_fnu_shape(wavelength_a: jnp.ndarray, temperature_k: jnp.ndarray) -> jnp.ndarray:
    """Return an arbitrary-normalized blackbody F_nu shape."""

    wavelength_m = jnp.maximum(wavelength_a, 1.0) * 1.0e-10
    nu = _C_M_PER_S / wavelength_m
    x = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * temperature_k), 1.0e-6, 700.0)
    return nu**3 / jnp.expm1(x)


def _blackbody_state(config: JointFitConfig, component: SedComponentConfig) -> Mapping[str, Any]:
    """Evaluate a simple blackbody point-source SED in filter space."""

    wavelengths = _filter_wavelengths(config)
    ensure_grahspj_context(config)
    rest_wave = (
        jnp.asarray(config.grahspj_context.rest_wave_jax, dtype=jnp.float64)
        if config.grahspj_context is not None
        else wavelengths
    )
    obs_wave = (
        jnp.asarray(config.grahspj_context.obs_wave_jax, dtype=jnp.float64)
        if config.grahspj_context is not None
        else wavelengths
    )
    if component.fit_temperature:
        log_temperature = numpyro.sample(
            f"{component.name}/log_temperature_k",
            dist.Normal(jnp.log(component.temperature_k), 0.25),
        )
        temperature = jnp.exp(log_temperature)
    else:
        temperature = jnp.asarray(component.temperature_k, dtype=jnp.float64)
    if component.fit_reference_flux:
        log_flux = numpyro.sample(
            f"{component.name}/log_reference_flux_mjy",
            dist.Normal(jnp.log(component.reference_flux_mjy), component.log_reference_flux_sigma),
        )
        reference_flux = jnp.exp(log_flux)
    else:
        reference_flux = jnp.asarray(component.reference_flux_mjy, dtype=jnp.float64)
    shape = _blackbody_fnu_shape(wavelengths, temperature)
    filter_names = [band.filter_name for band in config.image_bands]
    if component.reference_filter_name is not None and component.reference_filter_name in filter_names:
        reference_index = filter_names.index(component.reference_filter_name)
    else:
        reference_index = len(filter_names) // 2
    pred_fluxes = reference_flux * shape / jnp.maximum(shape[reference_index], 1.0e-300)
    zeros = jnp.zeros_like(pred_fluxes)
    return {
        "pred_fluxes": pred_fluxes,
        "agn_fluxes": zeros,
        "host_fluxes": zeros,
        "star_fluxes": pred_fluxes,
        "dust_fluxes": zeros,
        "nebular_fluxes": zeros,
        "rest_wave": rest_wave,
        "obs_wave": obs_wave,
        "total_obs_sed": jnp.zeros_like(obs_wave),
        "total_rest_sed": jnp.zeros_like(rest_wave),
        "kind": "star",
    }


def _component_states_from_trace(
    config: JointFitConfig,
    sampled: Mapping[str, Any] | None = None,
    *,
    include_components: bool = False,
) -> dict[str, Mapping[str, Any]]:
    """Evaluate all configured SED components."""

    ensure_grahspj_context(config)

    def _eval_all():
        states: dict[str, Mapping[str, Any]] = {}
        for component in config.resolved_sed_components:
            if component.kind == "agn" or component.kind in _GALAXY_KINDS:
                states[component.name] = _scoped_component_state(component, include_components=include_components)
            elif component.kind == "star":
                states[component.name] = _blackbody_state(config, component)
        return states

    if sampled is None:
        return _eval_all()
    return handlers.substitute(_eval_all, data=dict(sampled))()


def _combine_component_states(
    config: JointFitConfig,
    component_states: Mapping[str, Mapping[str, Any]],
    *,
    include_components: bool,
) -> dict[str, Any]:
    """Sum component SED states into grahspj-like aggregate state arrays."""

    template_state = next(iter(component_states.values()))
    pred_fluxes = jnp.zeros_like(template_state["pred_fluxes"])
    agn_fluxes = jnp.zeros_like(pred_fluxes)
    host_fluxes = jnp.zeros_like(pred_fluxes)
    star_fluxes = jnp.zeros_like(pred_fluxes)
    dust_fluxes = jnp.zeros_like(pred_fluxes)
    nebular_fluxes = jnp.zeros_like(pred_fluxes)
    component_by_name = {component.name: component for component in config.resolved_sed_components}
    component_fluxes: dict[str, jnp.ndarray] = {}
    for name, state in component_states.items():
        component = component_by_name[name]
        if isinstance(state, dict):
            state.setdefault("kind", component.kind)
        component_fluxes_by_filter = state["pred_fluxes"]
        component_fluxes[name] = component_fluxes_by_filter
        pred_fluxes = pred_fluxes + component_fluxes_by_filter
        if component.kind == "agn":
            agn_fluxes = agn_fluxes + component_fluxes_by_filter
        elif component.kind in _GALAXY_KINDS:
            host_fluxes = host_fluxes + component_fluxes_by_filter
            dust_fluxes = dust_fluxes + state.get("dust_fluxes", jnp.zeros_like(pred_fluxes))
            nebular_fluxes = nebular_fluxes + state.get("nebular_fluxes", jnp.zeros_like(pred_fluxes))
        elif component.kind == "star":
            star_fluxes = star_fluxes + component_fluxes_by_filter

    aggregate: dict[str, Any] = {
        "pred_fluxes": pred_fluxes,
        "agn_fluxes": agn_fluxes,
        "host_fluxes": host_fluxes,
        "star_fluxes": star_fluxes,
        "dust_fluxes": dust_fluxes,
        "nebular_fluxes": nebular_fluxes,
        "component_states": component_states,
        "component_fluxes": component_fluxes,
        "rest_wave": template_state.get("rest_wave", jnp.asarray([])),
        "obs_wave": template_state.get("obs_wave", jnp.asarray([])),
        "redshift_fit": template_state.get("redshift_fit", jnp.asarray(0.0)),
    }
    if include_components:
        for key in ("total_rest_sed", "agn_rest_sed", "host_rest_sed", "host_total_rest_sed", "dust_rest_sed", "nebular_rest_sed"):
            total = None
            for state in component_states.values():
                if key not in state:
                    continue
                total = state[key] if total is None else total + state[key]
            if total is not None:
                aggregate[key] = total
        for key in ("total_obs_sed", "agn_obs_sed", "host_obs_sed", "host_total_obs_sed", "dust_obs_sed", "nebular_obs_sed"):
            total = None
            for state in component_states.values():
                if key not in state:
                    continue
                total = state[key] if total is None else total + state[key]
            if total is not None:
                aggregate[key] = total
    return aggregate


def _add_joint_photometry_likelihood(config: JointFitConfig, state: Mapping[str, Any]) -> None:
    """Apply one shared photometry likelihood to the summed component SED."""

    ensure_grahspj_context(config)
    from grahspj.model import photometric_loglike

    context = config.grahspj_context
    cfg = context.fit_config
    zeros = jnp.zeros_like(state["pred_fluxes"])
    logl = photometric_loglike(
        pred_fluxes=state["pred_fluxes"],
        obs_fluxes=jnp.asarray(context.fluxes, dtype=jnp.float64),
        obs_errors=jnp.asarray(context.errors, dtype=jnp.float64),
        upper_limits=jnp.asarray(context.upper_limits, dtype=bool),
        data_mask=jnp.asarray(context.data_mask, dtype=bool),
        systematics_width=cfg.likelihood.systematics_width,
        intrinsic_scatter=jnp.asarray(cfg.likelihood.intrinsic_scatter_default, dtype=jnp.float64),
        student_t_df=cfg.likelihood.student_t_df,
        agn_component=state["agn_fluxes"],
        agn_bol_lum_w=jnp.asarray(0.0, dtype=jnp.float64),
        agn_nev=cfg.likelihood.agn_nev,
        variability_uncertainty=False,
        attenuation_model_uncertainty=False,
        transmitted_fraction=jnp.ones_like(state["pred_fluxes"]),
        lyman_break_uncertainty=False,
        filter_wavelength=jnp.asarray(context.filter_effective_wavelength_jax, dtype=jnp.float64),
        redshift=state.get("redshift_fit", context.fixed_redshift_jax),
    )
    del zeros
    numpyro.factor("joint_photometry_loglike", logl)
    numpyro.deterministic("pred_fluxes", state["pred_fluxes"])
    numpyro.deterministic("agn_fluxes", state["agn_fluxes"])
    numpyro.deterministic("host_fluxes", state["host_fluxes"])
    numpyro.deterministic("star_fluxes", state["star_fluxes"])


def _grahspj_fluxes_by_band(config: JointFitConfig, state: Mapping[str, Any]) -> dict[str, ComponentFluxes]:
    """Map grahspj component flux arrays to image-band count-scaled components."""

    filter_names = list(config.grahspj_context.fit_config.photometry.filter_names)
    index = {name: i for i, name in enumerate(filter_names)}
    agn_fluxes = state["agn_fluxes"]
    host_fluxes = state["host_fluxes"]
    star_fluxes = state.get("star_fluxes", jnp.zeros_like(agn_fluxes))
    out: dict[str, ComponentFluxes] = {}
    for band in config.image_bands:
        if band.filter_name not in index:
            raise KeyError(f"Image filter {band.filter_name!r} is not present in grahspj photometry.")
        if band.counts_per_mjy is None:
            raise ValueError(f"Image band {band.filter_name!r} needs counts_per_mjy for grahspj rendering.")
        i = index[band.filter_name]
        scale = jnp.asarray(band.counts_per_mjy, dtype=jnp.float64)
        out[band.filter_name] = ComponentFluxes(
            agn=agn_fluxes[i] * scale,
            host=host_fluxes[i] * scale,
            star=star_fluxes[i] * scale,
        )
    return out


def _scene_fluxes_by_band(config: JointFitConfig, state: Mapping[str, Any]) -> dict[str, dict[str, jnp.ndarray]]:
    """Map component SED flux arrays to per-scene image count fluxes."""

    filter_names = list(config.grahspj_context.fit_config.photometry.filter_names)
    index = {name: i for i, name in enumerate(filter_names)}
    component_fluxes = state["component_fluxes"]
    out: dict[str, dict[str, jnp.ndarray]] = {}
    for band in config.image_bands:
        if band.filter_name not in index:
            raise KeyError(f"Image filter {band.filter_name!r} is not present in grahspj photometry.")
        if band.counts_per_mjy is None:
            raise ValueError(f"Image band {band.filter_name!r} needs counts_per_mjy for grahspj rendering.")
        i = index[band.filter_name]
        scale = jnp.asarray(band.counts_per_mjy, dtype=jnp.float64)
        out[band.filter_name] = {
            scene.name: component_fluxes[scene.sed_component][i] * scale
            for scene in config.resolved_scene_components
        }
    return out


def _scene_param_name(scene: SceneComponentConfig, param: str) -> str:
    """Return the NumPyro site/parameter name for one scene component parameter."""

    return f"{scene.name}/{param}"


def _scene_param(params: Mapping[str, Any], scene: SceneComponentConfig, param: str, default: float):
    return params.get(_scene_param_name(scene, param), default)


def _sample_scene_parameters(config: JointFitConfig) -> dict[str, Any]:
    """Sample or fix all scene spatial parameters."""

    priors = config.image.priors
    sampled: dict[str, Any] = {}
    for scene in config.resolved_scene_components:
        center_sigma = float(scene.center_sigma_pix if scene.center_sigma_pix is not None else priors.center_sigma_pix)
        if scene.fit_position:
            sampled[_scene_param_name(scene, "center_x_pix")] = numpyro.sample(
                _scene_param_name(scene, "center_x_pix"),
                dist.Normal(scene.fixed_center_x_pix, center_sigma),
            )
            sampled[_scene_param_name(scene, "center_y_pix")] = numpyro.sample(
                _scene_param_name(scene, "center_y_pix"),
                dist.Normal(scene.fixed_center_y_pix, center_sigma),
            )
        else:
            sampled[_scene_param_name(scene, "center_x_pix")] = scene.fixed_center_x_pix
            sampled[_scene_param_name(scene, "center_y_pix")] = scene.fixed_center_y_pix
        if scene.kind != "sersic":
            continue
        reff_loc = float(scene.reff_arcsec_loc if scene.reff_arcsec_loc is not None else priors.reff_arcsec_loc)
        reff_sigma = float(scene.reff_arcsec_sigma if scene.reff_arcsec_sigma is not None else priors.reff_arcsec_sigma)
        n_loc = float(scene.n_sersic_loc if scene.n_sersic_loc is not None else priors.n_sersic_loc)
        n_sigma = float(scene.n_sersic_sigma if scene.n_sersic_sigma is not None else priors.n_sersic_sigma)
        ell_sigma = float(scene.ellipticity_sigma if scene.ellipticity_sigma is not None else priors.ellipticity_sigma)
        if scene.fit_shape:
            sampled[_scene_param_name(scene, "reff_arcsec")] = numpyro.sample(
                _scene_param_name(scene, "reff_arcsec"),
                dist.TruncatedNormal(reff_loc, reff_sigma, low=1.0e-3),
            )
            sampled[_scene_param_name(scene, "n_sersic")] = numpyro.sample(
                _scene_param_name(scene, "n_sersic"),
                dist.TruncatedNormal(n_loc, n_sigma, low=0.3, high=8.0),
            )
            sampled[_scene_param_name(scene, "e1")] = numpyro.sample(
                _scene_param_name(scene, "e1"),
                dist.Normal(scene.fixed_e1, ell_sigma),
            )
            sampled[_scene_param_name(scene, "e2")] = numpyro.sample(
                _scene_param_name(scene, "e2"),
                dist.Normal(scene.fixed_e2, ell_sigma),
            )
        else:
            sampled[_scene_param_name(scene, "reff_arcsec")] = scene.fixed_reff_arcsec
            sampled[_scene_param_name(scene, "n_sersic")] = scene.fixed_n_sersic
            sampled[_scene_param_name(scene, "e1")] = scene.fixed_e1
            sampled[_scene_param_name(scene, "e2")] = scene.fixed_e2
    return sampled


def render_band_model(
    band,
    fluxes: ComponentFluxes,
    *,
    reff_arcsec,
    n_sersic,
    e1,
    e2,
    center_x_pix,
    center_y_pix,
    background,
) -> dict[str, jnp.ndarray]:
    """Render AGN, host, and total image components for one band."""

    shape = tuple(band.image.shape)
    psf = psf_unit_flux(jnp.asarray(band.psf), shape, center_x_pix, center_y_pix)
    host_unit = sersic_ellipse_unit_flux(
        shape,
        float(band.pixel_scale),
        reff_arcsec,
        n_sersic,
        e1,
        e2,
        center_x_pix,
        center_y_pix,
    )
    agn = psf * jnp.asarray(fluxes.agn, dtype=jnp.float64)
    star = psf * jnp.asarray(fluxes.star, dtype=jnp.float64)
    host = host_unit * jnp.asarray(fluxes.host, dtype=jnp.float64)
    background_image = jnp.ones(shape, dtype=jnp.float64) * background
    total = agn + star + host + background_image
    return {"total": total, "agn": agn, "star": star, "host": host, "background": background_image}


def render_joint_model(
    config: JointFitConfig,
    params: Mapping[str, Any],
    *,
    fluxes_by_band: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Render all image bands for a parameter dictionary."""

    config.validate()
    if fluxes_by_band is None:
        state = _grahspj_state_from_trace(config, params, add_likelihood=False, include_components=False)
        fluxes_by_band = _scene_fluxes_by_band(config, state)
    rendered = {}
    sed_by_name = {component.name: component for component in config.resolved_sed_components}
    for i, band in enumerate(config.image_bands):
        shape = tuple(band.image.shape)
        background = params.get(f"background_{i}", config.image.background_default)
        background_image = jnp.ones(shape, dtype=jnp.float64) * background
        total = background_image
        agn = jnp.zeros(shape, dtype=jnp.float64)
        host = jnp.zeros(shape, dtype=jnp.float64)
        star = jnp.zeros(shape, dtype=jnp.float64)
        band_payload = fluxes_by_band[band.filter_name]
        if isinstance(band_payload, ComponentFluxes) or not isinstance(band_payload, Mapping):
            legacy = band_payload if isinstance(band_payload, ComponentFluxes) else coerce_component_fluxes(band_payload)
            rendered[band.filter_name] = render_band_model(
                band,
                legacy,
                reff_arcsec=params.get("host_reff_arcsec", config.image.fixed_reff_arcsec),
                n_sersic=params.get("host_n_sersic", config.image.fixed_n_sersic),
                e1=params.get("host_e1", config.image.fixed_e1),
                e2=params.get("host_e2", config.image.fixed_e2),
                center_x_pix=params.get("center_x_pix", config.image.fixed_center_x_pix),
                center_y_pix=params.get("center_y_pix", config.image.fixed_center_y_pix),
                background=background,
            )
            continue
        components: dict[str, jnp.ndarray] = {}
        for scene in config.resolved_scene_components:
            flux = jnp.asarray(band_payload[scene.name], dtype=jnp.float64)
            center_x = _scene_param(params, scene, "center_x_pix", scene.fixed_center_x_pix)
            center_y = _scene_param(params, scene, "center_y_pix", scene.fixed_center_y_pix)
            if scene.kind == "point":
                image = psf_unit_flux(jnp.asarray(band.psf), shape, center_x, center_y) * flux
            elif scene.kind == "sersic":
                unit = sersic_ellipse_unit_flux(
                    shape,
                    float(band.pixel_scale),
                    _scene_param(params, scene, "reff_arcsec", scene.fixed_reff_arcsec),
                    _scene_param(params, scene, "n_sersic", scene.fixed_n_sersic),
                    _scene_param(params, scene, "e1", scene.fixed_e1),
                    _scene_param(params, scene, "e2", scene.fixed_e2),
                    center_x,
                    center_y,
                )
                unit = convolve_fft_same(unit, jnp.asarray(band.psf))
                image = unit * flux
            else:  # pragma: no cover - validate catches this
                raise ValueError(f"Unsupported scene component kind {scene.kind!r}.")
            components[scene.name] = image
            total = total + image
            sed_kind = sed_by_name[scene.sed_component].kind
            if sed_kind == "agn":
                agn = agn + image
            elif sed_kind in _GALAXY_KINDS:
                host = host + image
            elif sed_kind == "star":
                star = star + image
        rendered[band.filter_name] = {
            "total": total,
            "agn": agn,
            "host": host,
            "star": star,
            "background": background_image,
            "components": components,
            **components,
        }
    return rendered


def jaguar_model(config: JointFitConfig) -> None:
    """NumPyro joint image model.

    A future grahspj bridge can sample SED parameters before
    `resolve_component_fluxes`; fixed fluxes are sufficient for renderer and
    likelihood validation.
    """

    config.validate()
    grahspj_state = _grahspj_state_from_trace(config, None, add_likelihood=True, include_components=False)
    fluxes_by_band = _scene_fluxes_by_band(config, grahspj_state)
    sampled = _sample_scene_parameters(config)

    for i, _band in enumerate(config.image_bands):
        if config.image.fit_background:
            sampled[f"background_{i}"] = numpyro.sample(
                f"background_{i}",
                dist.Normal(config.image.background_default, config.image.priors.background_sigma),
            )
        else:
            sampled[f"background_{i}"] = config.image.background_default

    rendered = render_joint_model(config, sampled, fluxes_by_band=fluxes_by_band)
    for band in config.image_bands:
        model = rendered[band.filter_name]["total"]
        numpyro.deterministic(f"model_{band.filter_name}", model)
        mask = jnp.ones_like(model, dtype=bool) if band.mask is None else jnp.asarray(band.mask, dtype=bool)
        data = jnp.asarray(band.image, dtype=jnp.float64)
        noise = jnp.asarray(band.noise, dtype=jnp.float64)
        resid = jnp.where(mask, (data - model) / noise, 0.0)
        if config.image.use_student_t:
            log_prob = dist.StudentT(config.image.student_t_df, 0.0, 1.0).log_prob(resid)
            numpyro.factor(
                f"pixels_{band.filter_name}",
                jnp.sum(jnp.where(mask, log_prob, 0.0)),
            )
        else:
            log_prob = dist.Normal(0.0, 1.0).log_prob(resid)
            numpyro.factor(
                f"pixels_{band.filter_name}",
                jnp.sum(jnp.where(mask, log_prob, 0.0)),
            )
