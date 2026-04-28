from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta

from .config import JointFitConfig
from .model import _image_fluxes_by_band, _initial_scene_flux_count, _scene_flux_param_name, _scene_fluxes_by_band, _grahspj_state_from_trace, ensure_grahspj_context, image_flux_state_from_fluxes, jaguar_model, render_joint_model
from .result import JaguarResult


def _float_init(value: Any) -> jnp.ndarray:
    """Coerce initial latent values to differentiable floating arrays."""

    return jnp.asarray(value, dtype=jnp.float64)


def _bounded_float_init(value: Any, *, low: float | None = None, high: float | None = None) -> jnp.ndarray:
    """Coerce an initial latent value and keep it inside finite prior support."""

    initial = float(value)
    if low is not None:
        initial = max(initial, float(low) + 1.0e-8)
    if high is not None:
        initial = min(initial, float(high) - 1.0e-8)
    return _float_init(initial)


def _initial_values(config: JointFitConfig) -> dict[str, Any]:
    values: dict[str, Any] = {}
    image = config.image
    for scene in config.resolved_scene_components:
        for band in config.image_bands:
            if not config.joint_grahspj_fitting:
                values[_scene_flux_param_name(scene, band.filter_name)] = _float_init(
                    np.log(_initial_scene_flux_count(config, scene, band))
                )
        if scene.fit_position:
            values[f"{scene.name}/center_x_pix"] = _float_init(scene.fixed_center_x_pix)
            values[f"{scene.name}/center_y_pix"] = _float_init(scene.fixed_center_y_pix)
        if scene.kind == "sersic" and scene.fit_shape:
            values[f"{scene.name}/reff_arcsec"] = _bounded_float_init(
                scene.fixed_reff_arcsec,
                low=scene.min_reff_arcsec,
                high=scene.max_reff_arcsec,
            )
            values[f"{scene.name}/n_sersic"] = _float_init(scene.fixed_n_sersic)
            values[f"{scene.name}/e1"] = _float_init(scene.fixed_e1)
            values[f"{scene.name}/e2"] = _float_init(scene.fixed_e2)
    for i, _band in enumerate(config.image_bands):
        if image.fit_background:
            values[f"background_{i}"] = _float_init(image.background_default)
    return values


def _init_to_values_then_median_site(site=None, *, values: dict[str, Any], median_init):
    """Initialize selected image parameters explicitly and all other sites at prior medians."""

    if site is None:
        return partial(_init_to_values_then_median_site, values=values, median_init=median_init)
    if site["type"] == "sample" and not site["is_observed"] and site["name"] in values:
        return values[site["name"]]
    return median_init(site)


def _init_to_values_then_median(values: dict[str, Any]):
    """Return a NumPyro-compatible init strategy."""

    return partial(_init_to_values_then_median_site, values=values, median_init=init_to_median(num_samples=15))


def fit_map(
    config: JointFitConfig,
    *,
    seed: int = 11,
    steps: int = 1000,
    learning_rate: float = 5.0e-3,
    progress_bar: bool = False,
) -> JaguarResult:
    """Run an Optax MAP fit."""

    config.validate()
    if config.joint_grahspj_fitting:
        ensure_grahspj_context(config)
    rng_key = jax.random.PRNGKey(seed)
    guide = AutoDelta(jaguar_model, init_loc_fn=_init_to_values_then_median(_initial_values(config)))
    svi = SVI(jaguar_model, guide, optax.adam(learning_rate), Trace_ELBO())
    svi_result = svi.run(rng_key, int(steps), config, progress_bar=progress_bar)
    params = svi_result.params
    map_params = guide.median(params)
    if config.joint_grahspj_fitting:
        grahspj_state = _grahspj_state_from_trace(config, map_params, add_likelihood=False, include_components=True)
        fluxes_by_band = _scene_fluxes_by_band(config, grahspj_state)
    else:
        fluxes_by_band = _image_fluxes_by_band(config, map_params)
        grahspj_state = image_flux_state_from_fluxes(config, fluxes_by_band)
    rendered = render_joint_model(config, map_params, fluxes_by_band=fluxes_by_band)
    return JaguarResult(
        config=config,
        map_params={k: np.asarray(v) for k, v in map_params.items()},
        samples=None,
        rendered=rendered,
        grahspj_state={k: np.asarray(v) for k, v in grahspj_state.items()},
    )


def fit(
    config: JointFitConfig,
    *,
    fit_method: str = "optax+nuts",
    seed: int = 11,
    map_steps: int = 1000,
    learning_rate: float = 5.0e-3,
    nuts_warmup: int = 500,
    nuts_samples: int = 500,
    progress_bar: bool = True,
) -> JaguarResult:
    """Fit the joint image model."""

    if fit_method == "map_only":
        return fit_map(config, seed=seed, steps=map_steps, learning_rate=learning_rate, progress_bar=progress_bar)
    if fit_method != "optax+nuts":
        raise ValueError("fit_method must be 'map_only' or 'optax+nuts'.")
    map_result = fit_map(config, seed=seed, steps=map_steps, learning_rate=learning_rate, progress_bar=progress_bar)
    init_values = {k: jnp.asarray(v) for k, v in (map_result.map_params or {}).items()}
    kernel = NUTS(jaguar_model, init_strategy=_init_to_values_then_median(init_values))
    mcmc = MCMC(kernel, num_warmup=nuts_warmup, num_samples=nuts_samples, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed + 1), config)
    samples = mcmc.get_samples()
    median_params = {k: jnp.nanmedian(v, axis=0) for k, v in samples.items()}
    if config.joint_grahspj_fitting:
        grahspj_state = _grahspj_state_from_trace(config, median_params, add_likelihood=False, include_components=True)
        fluxes_by_band = _scene_fluxes_by_band(config, grahspj_state)
    else:
        fluxes_by_band = _image_fluxes_by_band(config, median_params)
        grahspj_state = image_flux_state_from_fluxes(config, fluxes_by_band)
    rendered = render_joint_model(config, median_params, fluxes_by_band=fluxes_by_band)
    return JaguarResult(
        config=config,
        map_params={k: np.asarray(v) for k, v in median_params.items()},
        samples={k: np.asarray(v) for k, v in samples.items()},
        rendered=rendered,
        grahspj_state={k: np.asarray(v) for k, v in grahspj_state.items()},
    )
