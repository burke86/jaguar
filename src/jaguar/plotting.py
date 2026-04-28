from __future__ import annotations

import numpy as np

from .detection import SourceDetectionConfig, build_detection_image


def _positive_log_image(image: np.ndarray, offset: float) -> np.ndarray:
    """Return an image shifted for stable logarithmic display."""

    finite = np.asarray(image, dtype=float)
    finite = np.where(np.isfinite(finite), finite, np.nan)
    return finite + offset


def plot_fit(
    result,
    filter_name: str | None = None,
    *,
    log_scale: bool = True,
    shared_vmin: float | None = None,
    shared_vmax: float | None = None,
    log_floor_fraction: float = 1.0e-4,
):
    """Plot data, model, components, and residual for one image band."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if filter_name is None:
        filter_name = result.config.image_bands[0].filter_name
    band = next(b for b in result.config.image_bands if b.filter_name == filter_name)
    comp = result.rendered[filter_name]
    data = np.asarray(band.image)
    model = np.asarray(comp["total"])
    resid = (data - model) / np.asarray(band.noise)
    panels = [
        ("Data", data),
        ("Model", model),
        ("AGN", np.asarray(comp["agn"])),
        ("Host", np.asarray(comp["host"])),
        ("Residual", resid),
    ]
    science_images = [image for title, image in panels if title != "Residual"]
    science_min = min(float(np.nanmin(image)) for image in science_images)
    offset = max(0.0, -science_min)
    shifted_science = [image + offset for image in science_images]
    if shared_vmax is None:
        shared_vmax = max(float(np.nanpercentile(image, 99.5)) for image in shifted_science)
    if shared_vmin is None:
        shared_vmin = max(float(shared_vmax) * float(log_floor_fraction), 1.0e-30)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2), constrained_layout=True)
    for ax, (title, image) in zip(axes, panels, strict=True):
        if log_scale and title != "Residual":
            display = np.clip(_positive_log_image(image, offset), shared_vmin, shared_vmax)
            im = ax.imshow(display, origin="lower", cmap="viridis", norm=LogNorm(vmin=shared_vmin, vmax=shared_vmax))
        else:
            vmax = np.nanpercentile(np.abs(image), 99.0)
            im = ax.imshow(image, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.text(
            0.04,
            0.96,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 3},
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    return fig, axes


def plot_detection(
    image_bands,
    detections,
    config: SourceDetectionConfig | None = None,
    *,
    detection_image: np.ndarray | None = None,
    figsize: tuple[float, float] = (5.0, 5.0),
    cmap: str = "viridis",
    log_floor_fraction: float = 1.0e-4,
    title: str = "Combined detection image",
):
    """Plot the combined source-detection image and non-central detections."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle

    cfg = SourceDetectionConfig() if config is None else config
    image = build_detection_image(image_bands, cfg) if detection_image is None else np.asarray(detection_image, dtype=float)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        display = np.ones_like(image, dtype=float)
        vmin = 1.0
        vmax = 1.0
    else:
        offset = max(0.0, -float(np.nanmin(finite)))
        display = image + offset
        positive = display[np.isfinite(display) & (display > 0.0)]
        vmax = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
        vmin = max(vmax * float(log_floor_fraction), float(np.nanmin(positive)) if positive.size else 1.0e-30, 1.0e-30)
        display = np.clip(display, vmin, max(vmax, vmin))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display, origin="lower", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin)))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cx = (image.shape[1] - 1) / 2.0
    cy = (image.shape[0] - 1) / 2.0
    if cfg.target_exclusion_radius_pix > 0.0:
        ax.add_patch(
            Circle(
                (cx, cy),
                float(cfg.target_exclusion_radius_pix),
                fill=False,
                edgecolor="white",
                linewidth=1.3,
                linestyle="--",
                alpha=0.9,
            )
        )
    for source in detections:
        if np.hypot(float(source.x_pix) - cx, float(source.y_pix) - cy) <= float(cfg.target_exclusion_radius_pix):
            continue
        color = "cyan" if source.classification == "galaxy" else "lime"
        ax.plot(source.x_pix, source.y_pix, "o", ms=10, mfc="none", mec=color, mew=1.6)
        ax.text(source.x_pix + 1, source.y_pix + 1, source.classification, color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig, ax


class _GrahspjPlotAdapter:
    """Small fitter-shaped adapter for grahspj's SED plotting helper."""

    def __init__(self, result, component: str = "total"):
        if result.grahspj_state is None:
            raise ValueError("No grahspj SED state is available on this JAGUAR result.")
        if result.config.grahspj_config is None:
            raise ValueError("No grahspj config is available on this JAGUAR result.")
        if result.config.grahspj_context is None:
            from .model import ensure_grahspj_context

            ensure_grahspj_context(result.config)
        self.config = result.config.grahspj_config
        self.context = result.config.grahspj_context
        self._state = _select_sed_component_state(result.grahspj_state, component)

    def predict(self, posterior: str = "latest"):
        """Return a grahspj-like predictive dictionary."""

        del posterior
        pred = {}
        for key, value in self._state.items():
            arr = np.asarray(value)
            pred[key] = arr[None, ...] if arr.ndim > 0 else arr
        return pred


def plot_sed(
    result,
    *,
    component: str = "total",
    output_path=None,
    posterior: str = "latest",
    show: bool = False,
    annotate_band_names: bool = True,
):
    """Plot the total SED or one named JAGUAR SED component."""

    from grahspj.plotting import plot_fit_sed

    return plot_fit_sed(
        _GrahspjPlotAdapter(result, component=component),
        output_path=output_path,
        posterior=posterior,
        show=show,
        annotate_band_names=annotate_band_names,
    )


def _select_sed_component_state(state, component: str):
    """Return a grahspj-like state for the total SED or one named component."""

    if component == "total":
        return state
    component_states = state.get("component_states", {})
    if component not in component_states:
        available = ", ".join(["total", *sorted(component_states)])
        raise KeyError(f"Unknown SED component {component!r}. Available components: {available}.")
    selected = dict(component_states[component])
    out = dict(state)
    pred_fluxes = np.asarray(selected["pred_fluxes"])
    out["pred_fluxes"] = pred_fluxes
    out["total_obs_sed"] = selected.get("total_obs_sed", np.zeros_like(np.asarray(state.get("obs_wave", []), dtype=float)))
    out["total_rest_sed"] = selected.get("total_rest_sed", np.zeros_like(np.asarray(state.get("rest_wave", []), dtype=float)))
    zero_fluxes = np.zeros_like(pred_fluxes)
    for key in ("agn_fluxes", "host_fluxes", "dust_fluxes", "nebular_fluxes"):
        out[key] = zero_fluxes
    for key in ("agn_obs_sed", "host_obs_sed", "dust_obs_sed", "nebular_obs_sed"):
        out[key] = np.zeros_like(np.asarray(out["total_obs_sed"], dtype=float))
    for key in ("agn_rest_sed", "host_rest_sed", "dust_rest_sed", "nebular_rest_sed"):
        out[key] = np.zeros_like(np.asarray(out["total_rest_sed"], dtype=float))

    kind = selected.get("kind")
    if kind is None:
        if np.any(np.asarray(selected.get("agn_fluxes", zero_fluxes), dtype=float)):
            kind = "agn"
        elif np.any(np.asarray(selected.get("star_fluxes", zero_fluxes), dtype=float)):
            kind = "star"
        else:
            kind = "host"
    if kind == "agn":
        out["agn_fluxes"] = pred_fluxes
        out["agn_obs_sed"] = out["total_obs_sed"]
        out["agn_rest_sed"] = out["total_rest_sed"]
    elif kind == "star":
        out["agn_fluxes"] = pred_fluxes
        out["agn_obs_sed"] = out["total_obs_sed"]
        out["agn_rest_sed"] = out["total_rest_sed"]
    else:
        out["host_fluxes"] = pred_fluxes
        out["host_obs_sed"] = out["total_obs_sed"]
        out["host_rest_sed"] = out["total_rest_sed"]
    return out
