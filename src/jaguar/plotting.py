from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import numpy as np
import warnings

from .detection import SourceDetectionConfig


def _positive_log_image(image: np.ndarray, offset: float) -> np.ndarray:
    """Return an image shifted for stable logarithmic display."""

    finite = np.asarray(image, dtype=float)
    finite = np.where(np.isfinite(finite), finite, np.nan)
    return finite + offset


def _radial_surface_brightness_profile(
    image: np.ndarray,
    pixel_scale: float,
    *,
    noise: np.ndarray | None = None,
    zeropoint: float | None = None,
    center_xy: tuple[float, float] | None = None,
    bin_width_arcsec: float | None = None,
    snr_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an azimuthal mean surface-brightness profile in mag/arcsec^2."""

    image = np.asarray(image, dtype=float)
    noise = None if noise is None else np.asarray(noise, dtype=float)
    ny, nx = image.shape
    if center_xy is None:
        center_xy = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    cx, cy = center_xy
    yy, xx = np.indices(image.shape, dtype=float)
    radius = np.hypot(xx - cx, yy - cy) * float(pixel_scale)
    max_radius = float(np.nanmax(radius))
    if bin_width_arcsec is None:
        bin_width_arcsec = max(float(pixel_scale), max_radius / 25.0)
    edges = np.arange(0.0, max_radius + float(bin_width_arcsec), float(bin_width_arcsec))
    if edges.size < 2:
        edges = np.asarray([0.0, max_radius + float(pixel_scale)])
    area = max(float(pixel_scale) ** 2, 1.0e-30)
    sb = image / area
    sb_noise = None if noise is None else noise / area
    radii: list[float] = []
    values: list[float] = []
    errors: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (radius >= lo) & (radius < hi) & np.isfinite(sb)
        if not np.any(mask):
            continue
        n_pix = int(np.count_nonzero(mask))
        value = float(np.nanmean(sb[mask]))
        error = np.nan
        if sb_noise is not None:
            noise_values = sb_noise[mask]
            noise_values = noise_values[np.isfinite(noise_values)]
            if noise_values.size:
                error = float(np.sqrt(np.sum(noise_values**2)) / max(n_pix, 1))
        if zeropoint is not None:
            valid = np.isfinite(value) and value > 0.0
            if snr_floor is not None and np.isfinite(error) and error > 0.0:
                valid = valid and value > float(snr_floor) * error
            if valid:
                linear_value = value
                value = float(zeropoint) - 2.5 * np.log10(linear_value)
                if np.isfinite(error) and error > 0.0:
                    error = float(2.5 / np.log(10.0) * error / linear_value)
                else:
                    error = np.nan
            else:
                value = np.nan
                error = np.nan
        radii.append(0.5 * (lo + hi))
        values.append(value)
        errors.append(error)
    return np.asarray(radii), np.asarray(values), np.asarray(errors)


def _plot_profile(ax, radius: np.ndarray, profile: np.ndarray, *args, **kwargs):
    mask = np.isfinite(radius) & np.isfinite(profile)
    if np.any(mask):
        ax.plot(radius[mask], profile[mask], *args, **kwargs)


def _errorbar_profile(ax, radius: np.ndarray, profile: np.ndarray, error: np.ndarray, *args, **kwargs):
    mask = np.isfinite(radius) & np.isfinite(profile) & np.isfinite(error)
    if np.any(mask):
        ax.errorbar(radius[mask], profile[mask], yerr=error[mask], *args, **kwargs)


def _profile_host_reff_arcsec(result) -> float | None:
    """Return the main host effective radius for profile plotting."""

    scene_components = getattr(result.config, "resolved_scene_components", [])
    host_component = None
    for component in scene_components:
        if getattr(component, "kind", None) != "sersic":
            continue
        if getattr(component, "sed_component", None) == "host":
            host_component = component
            break
        if host_component is None:
            host_component = component
    if host_component is None:
        return None

    name = getattr(host_component, "name", "")
    key = f"{name}/reff_arcsec"
    value = getattr(result, "map_params", {}).get(key, getattr(host_component, "fixed_reff_arcsec", None))
    if value is None:
        return None
    reff = float(np.asarray(value))
    if not np.isfinite(reff) or reff <= 0.0:
        return None
    return reff


def _clip_profiles_to_radius(radius: np.ndarray, max_radius: float | None, *profiles: np.ndarray):
    if max_radius is None:
        return radius, profiles
    mask = np.asarray(radius) <= float(max_radius)
    if not np.any(mask):
        return radius, profiles
    return radius[mask], tuple(profile[mask] for profile in profiles)


def _host_component_images(result, comp: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Return rendered host scene-component images when available."""

    components = comp.get("components")
    if not isinstance(components, Mapping) or not components:
        host = comp.get("host")
        return [("Host", np.asarray(host))] if host is not None else []

    sed_by_name = {sed.name: sed for sed in result.config.resolved_sed_components}
    host_images: list[tuple[str, np.ndarray]] = []
    for scene in result.config.resolved_scene_components:
        if scene.name.startswith("det_"):
            continue
        sed_component = sed_by_name.get(scene.sed_component)
        if sed_component is None or sed_component.kind not in {"host", "galaxy"}:
            continue
        if scene.name not in components:
            continue
        host_images.append((scene.name, np.asarray(components[scene.name])))
    if host_images:
        return host_images
    host = comp.get("host")
    return [("Host", np.asarray(host))] if host is not None else []


def plot_fit(
    result,
    filter_name: str | None = None,
    *,
    log_scale: bool = True,
    shared_vmin: float | None = None,
    shared_vmax: float | None = None,
    log_floor_fraction: float = 1.0e-4,
    profile_radius_factor: float | None = 5.0,
    profile_snr_floor: float | None = 1.0,
):
    """Plot data, model, components, and residual for one image band."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if filter_name is None:
        filter_name = result.config.image_bands[0].filter_name
    band = next(b for b in result.config.image_bands if b.filter_name == filter_name)
    comp = result.rendered[filter_name]
    data = np.asarray(band.image)
    model = np.asarray(comp["total"])
    agn_image = np.asarray(comp["agn"])
    host_component_images = _host_component_images(result, comp)
    background_image = np.asarray(comp.get("background", np.zeros_like(model)))
    source_model = model - background_image
    data_minus_point = data - agn_image
    resid = (data - model) / np.asarray(band.noise)
    panels = [
        ("Data", data),
        ("Model", model),
        ("Data - Point Source", data_minus_point),
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
    fig = plt.figure(figsize=(3.0 * (len(panels) + 1), 4.0), constrained_layout=False)
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.08)
    image_axes = [fig.add_subplot(grid[:, i]) for i in range(len(panels))]
    profile_grid = grid[0, -1].subgridspec(2, 1, height_ratios=[3.4, 0.8], hspace=0.0)
    profile_ax = fig.add_subplot(profile_grid[0, 0])
    profile_resid_ax = fig.add_subplot(profile_grid[1, 0], sharex=profile_ax)
    science_im = None
    residual_im = None
    for ax, (title, image) in zip(image_axes, panels, strict=True):
        if log_scale and title != "Residual":
            display = np.clip(_positive_log_image(image, offset), shared_vmin, shared_vmax)
            im = ax.imshow(display, origin="lower", cmap="viridis", norm=LogNorm(vmin=shared_vmin, vmax=shared_vmax))
            science_im = im
        else:
            vmax = np.nanpercentile(np.abs(image), 99.0)
            im = ax.imshow(image, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            residual_im = im
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
    if science_im is not None:
        cax = inset_axes(
            image_axes[2],
            width="5%",
            height="42%",
            loc="lower left",
            bbox_to_anchor=(0.82, 0.08, 1.0, 1.0),
            bbox_transform=image_axes[2].transAxes,
            borderpad=0.0,
        )
        cbar = fig.colorbar(science_im, cax=cax)
        cbar.set_label("counts", color="white", fontsize=7, labelpad=2)
        cbar.ax.tick_params(labelsize=7, length=2, colors="white")
        cbar.ax.yaxis.label.set_color("white")
        cbar.outline.set_edgecolor("white")
    if residual_im is not None:
        cax = inset_axes(
            image_axes[3],
            width="5%",
            height="42%",
            loc="lower left",
            bbox_to_anchor=(0.82, 0.08, 1.0, 1.0),
            bbox_transform=image_axes[3].transAxes,
            borderpad=0.0,
        )
        cbar = fig.colorbar(residual_im, cax=cax)
        cbar.set_label(r"$(data-model)/\sigma$", color="white", fontsize=7, labelpad=2)
        cbar.ax.tick_params(labelsize=7, length=2, colors="white")
        cbar.ax.yaxis.label.set_color("white")
        cbar.outline.set_edgecolor("white")

    fig.canvas.draw()
    image_box = image_axes[0].get_position()
    profile_box = profile_ax.get_position()
    profile_height_ratio = 3.4
    residual_height_ratio = 0.8
    total_ratio = profile_height_ratio + residual_height_ratio
    residual_height = image_box.height * residual_height_ratio / total_ratio
    profile_height = image_box.height - residual_height
    profile_resid_ax.set_position([profile_box.x0, image_box.y0, image_box.width, residual_height])
    profile_ax.set_position([profile_box.x0, image_box.y0 + residual_height, image_box.width, profile_height])

    assumed_zeropoint = band.zeropoint is None
    zeropoint = band.zeropoint if band.zeropoint is not None else 22.5
    if assumed_zeropoint:
        warnings.warn(
            f"Band {band.filter_name!r} has no zeropoint; assuming AB zeropoint 22.5 for surface-brightness profile.",
            RuntimeWarning,
            stacklevel=2,
        )
    radius, data_profile, data_profile_error = _radial_surface_brightness_profile(
        data,
        band.pixel_scale,
        noise=np.asarray(band.noise),
        zeropoint=zeropoint,
        snr_floor=profile_snr_floor,
    )
    _radius, model_profile, _model_error = _radial_surface_brightness_profile(source_model, band.pixel_scale, zeropoint=zeropoint)
    _radius, agn_profile, _agn_error = _radial_surface_brightness_profile(agn_image, band.pixel_scale, zeropoint=zeropoint)
    host_profiles = [
        (name, _radial_surface_brightness_profile(image, band.pixel_scale, zeropoint=zeropoint)[1])
        for name, image in host_component_images
    ]
    profile_radius_max = None
    host_reff = _profile_host_reff_arcsec(result)
    if profile_radius_factor is not None and host_reff is not None:
        profile_radius_max = float(profile_radius_factor) * host_reff
    radius, profiles = _clip_profiles_to_radius(
        radius,
        profile_radius_max,
        data_profile,
        data_profile_error,
        model_profile,
        agn_profile,
        *(profile for _name, profile in host_profiles),
    )
    data_profile, data_profile_error, model_profile, agn_profile, *clipped_host_profiles = profiles
    host_profiles = [
        (name, profile)
        for (name, _profile), profile in zip(host_profiles, clipped_host_profiles, strict=False)
    ]
    _errorbar_profile(
        profile_ax,
        radius,
        data_profile,
        data_profile_error,
        fmt="o",
        ms=3,
        color="black",
        ecolor="0.35",
        elinewidth=0.8,
        capsize=1.5,
        label="Data",
    )
    _plot_profile(profile_ax, radius, model_profile, "-", lw=1.8, color="tab:blue", label="Model")
    _plot_profile(profile_ax, radius, agn_profile, "--", lw=1.5, color="tab:orange", label="AGN")
    host_colors = ["tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    for index, (name, host_profile) in enumerate(host_profiles):
        label = "Host" if name == "Host" else f"Host: {name}"
        _plot_profile(
            profile_ax,
            radius,
            host_profile,
            "-.",
            lw=1.5,
            color=host_colors[index % len(host_colors)],
            label=label,
        )
    if profile_radius_max is not None:
        profile_ax.set_xlim(0.0, profile_radius_max)
    profile_ax.invert_yaxis()
    profile_ax.set_ylabel(r"$\mu$ (mag arcsec$^{-2}$)", fontsize=10)
    profile_ax.yaxis.set_label_position("right")
    profile_ax.yaxis.tick_right()
    if assumed_zeropoint:
        profile_ax.text(
            0.03,
            0.03,
            "Assumed ZP=22.5",
            transform=profile_ax.transAxes,
            ha="left",
            va="bottom",
            color="crimson",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "crimson", "pad": 2},
        )
    profile_ax.legend(fontsize=8, frameon=False)
    profile_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    if radius.size and model_profile.size:
        n = min(radius.size, data_profile.size, data_profile_error.size, model_profile.size)
        delta = data_profile[:n] - model_profile[:n]
        profile_resid_ax.axhline(0.0, color="0.3", lw=1.0)
        _errorbar_profile(
            profile_resid_ax,
            radius[:n],
            delta,
            data_profile_error[:n],
            fmt="o",
            ms=2.8,
            color="black",
            ecolor="0.35",
            elinewidth=0.8,
            capsize=1.5,
        )
    profile_resid_ax.set_xlabel("Radius (arcsec)")
    profile_resid_ax.set_ylabel(r"$\Delta\mu$", fontsize=10)
    profile_resid_ax.yaxis.set_label_position("right")
    profile_resid_ax.yaxis.tick_right()
    axes = [*image_axes, profile_ax, profile_resid_ax]
    return fig, axes


def _log_display_image(image: np.ndarray, log_floor_fraction: float) -> tuple[np.ndarray, float, float]:
    finite = np.asarray(image, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.ones_like(image, dtype=float), 1.0, 1.0
    offset = max(0.0, -float(np.nanmin(finite)))
    display = np.asarray(image, dtype=float) + offset
    positive = display[np.isfinite(display) & (display > 0.0)]
    vmax = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
    vmin = max(vmax * float(log_floor_fraction), float(np.nanmin(positive)) if positive.size else 1.0e-30, 1.0e-30)
    return np.clip(display, vmin, max(vmax, vmin)), vmin, max(vmax, vmin)


def plot_psf_candidates(
    image: np.ndarray,
    candidates,
    *,
    target_pixel: tuple[float, float] | None = None,
    psf_size: int = 25,
    target_exclusion_radius_pix: float = 20.0,
    figsize: tuple[float, float] = (5.5, 5.0),
    cmap: str = "viridis",
    log_floor_fraction: float = 1.0e-4,
    title: str = "Empirical PSF candidates",
):
    """Plot a PSF search image with accepted empirical PSF sources circled."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.lines import Line2D

    image = np.asarray(image, dtype=float)
    display, vmin, vmax = _log_display_image(image, log_floor_fraction)
    if target_pixel is None:
        target_pixel = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    tx, ty = target_pixel
    half = float(psf_size) / 2.0

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(display, origin="lower", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if target_exclusion_radius_pix > 0.0:
        ax.add_patch(
            Circle(
                (tx, ty),
                float(target_exclusion_radius_pix),
                fill=False,
                edgecolor="white",
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
            )
        )
    ax.plot(tx, ty, "+", color="white", ms=8, mew=1.4)
    has_gaia_star = False
    for idx, candidate in enumerate(candidates, start=1):
        x = float(candidate.x_pix)
        y = float(candidate.y_pix)
        is_gaia_star = bool(getattr(candidate, "is_gaia_star", False))
        has_gaia_star = has_gaia_star or is_gaia_star
        edgecolor = "red" if is_gaia_star else "cyan"
        ax.add_patch(Circle((x, y), max(4.0, half * 0.25), fill=False, edgecolor=edgecolor, linewidth=1.5))
        ax.add_patch(Rectangle((x - half, y - half), 2.0 * half, 2.0 * half, fill=False, edgecolor=edgecolor, linewidth=0.8, alpha=0.65))
        ax.text(
            x + 2.0,
            y + 2.0,
            str(idx),
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
        )
    if has_gaia_star:
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="red", markerfacecolor="none", linestyle="none", label="Gaia star"),
            ],
            loc="upper right",
            fontsize=8,
            frameon=True,
        )
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig, ax


def plot_empirical_psf_selection(
    result,
    *,
    band: str | None = None,
    show_all_candidates: bool = False,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    log_floor_fraction: float = 1.0e-4,
):
    """Plot empirical PSF candidates selected by ``build_empirical_psfs_for_bands``."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    band_codes = [band] if band is not None else list(result.bands)
    band_results = [result.bands[band_code] for band_code in band_codes]
    if not band_results:
        raise ValueError("No empirical PSF bands are available to plot.")

    selected_index: dict[tuple[str, float, float], int] = {}
    for group_index, group in enumerate(result.common_star_groups, start=1):
        for group_band, candidate in group.items():
            selected_index[(group_band, float(candidate.x_pix), float(candidate.y_pix))] = group_index

    n_bands = len(band_results)
    if figsize is None:
        figsize = (4.8 * n_bands, 4.6)
    fig, axes = plt.subplots(1, n_bands, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["cyan"])
    psf_size = int(result.config.psf_size)
    half = float(psf_size) / 2.0
    has_gaia_star = False
    has_all_candidates = False

    for ax, band_result in zip(axes, band_results, strict=True):
        image = np.asarray(band_result.search_image, dtype=float)
        display, vmin, vmax = _log_display_image(image, log_floor_fraction)
        im = ax.imshow(display, origin="lower", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(f"PSF candidates: {band_result.band_code}")
        ax.set_xticks([])
        ax.set_yticks([])
        tx, ty = band_result.search_target_pixel
        radius = float(result.config.target_exclusion_radius_pix)
        if radius > 0.0:
            ax.add_patch(Circle((tx, ty), radius, fill=False, edgecolor="white", linewidth=1.2, linestyle="--", alpha=0.9))
        ax.plot(tx, ty, "+", color="white", ms=8, mew=1.4)

        selected_keys = {(float(candidate.x_pix), float(candidate.y_pix)) for candidate in band_result.selected_candidates}
        if show_all_candidates:
            for candidate in band_result.candidates:
                key = (float(candidate.x_pix), float(candidate.y_pix))
                if key in selected_keys:
                    continue
                has_all_candidates = True
                ax.add_patch(Circle(key, max(4.0, half * 0.25), fill=False, edgecolor="cyan", linewidth=0.9, alpha=0.35))

        for local_index, candidate in enumerate(band_result.selected_candidates, start=1):
            x = float(candidate.x_pix)
            y = float(candidate.y_pix)
            group_index = selected_index.get((band_result.band_code, x, y), local_index)
            color = colors[(group_index - 1) % len(colors)]
            is_gaia_star = bool(getattr(candidate, "is_gaia_star", False))
            has_gaia_star = has_gaia_star or is_gaia_star
            edgecolor = "red" if is_gaia_star else color
            ax.add_patch(Circle((x, y), max(4.0, half * 0.25), fill=False, edgecolor=edgecolor, linewidth=1.7))
            ax.add_patch(Rectangle((x - half, y - half), 2.0 * half, 2.0 * half, fill=False, edgecolor=edgecolor, linewidth=0.9, alpha=0.75))
            ax.text(
                x + 2.0,
                y + 2.0,
                str(group_index),
                color="white",
                fontsize=8,
                fontweight="bold",
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
            )
        fig.colorbar(im, ax=ax, fraction=0.046)

    handles = [Line2D([0], [0], marker="o", color=colors[0], markerfacecolor="none", linestyle="none", label="Selected PSF star")]
    if show_all_candidates and has_all_candidates:
        handles.append(Line2D([0], [0], marker="o", color="cyan", markerfacecolor="none", linestyle="none", alpha=0.35, label="Other candidate"))
    if has_gaia_star:
        handles.append(Line2D([0], [0], marker="o", color="red", markerfacecolor="none", linestyle="none", label="Gaia star"))
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    return fig, axes


def plot_config(
    image_bands,
    detections,
    config: SourceDetectionConfig | None = None,
    *,
    detection_image: np.ndarray | None = None,
    filter_name: str | None = None,
    figsize: tuple[float, float] = (12.5, 4.2),
    cmap: str = "viridis",
    log_floor_fraction: float = 1.0e-4,
    title: str = "Target image",
):
    """Plot each band's target image, PSF image, and PSF radial profile."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle

    cfg = SourceDetectionConfig() if config is None else config
    bands = list(image_bands)
    if not bands:
        raise ValueError("At least one image band is required for plot_config.")
    if filter_name is not None:
        bands = [next(b for b in bands if b.filter_name == filter_name)]
    n_bands = len(bands)
    row_height = figsize[1]
    fig, axes = plt.subplots(n_bands, 3, figsize=(figsize[0], row_height * n_bands), constrained_layout=True, squeeze=False)

    for row, band in enumerate(bands):
        image = np.asarray(band.image, dtype=float)
        display, vmin, vmax = _log_display_image(image, log_floor_fraction)
        image_ax = axes[row, 0]
        im = image_ax.imshow(display, origin="lower", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        image_ax.set_title(f"{title}: {band.filter_name}")
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        cx = (image.shape[1] - 1) / 2.0
        cy = (image.shape[0] - 1) / 2.0
        if cfg.target_exclusion_radius_pix > 0.0:
            image_ax.add_patch(
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
            image_ax.plot(source.x_pix, source.y_pix, "o", ms=10, mfc="none", mec=color, mew=1.6)
            image_ax.text(source.x_pix + 1, source.y_pix + 1, source.classification, color="white", fontsize=8)
        fig.colorbar(im, ax=image_ax, fraction=0.046)

        psf = np.asarray(band.psf, dtype=float)
        psf_display, psf_vmin, psf_vmax = _log_display_image(psf, log_floor_fraction)
        psf_ax = axes[row, 1]
        psf_im = psf_ax.imshow(psf_display, origin="lower", cmap=cmap, norm=LogNorm(vmin=psf_vmin, vmax=psf_vmax))
        psf_ax.set_title(f"PSF: {band.filter_name}")
        psf_ax.set_xticks([])
        psf_ax.set_yticks([])
        fig.colorbar(psf_im, ax=psf_ax, fraction=0.046)

        psf_profile_ax = axes[row, 2]
        psf_flux = float(np.nansum(np.clip(psf, 0.0, np.inf)))
        psf_unit = np.clip(psf, 0.0, np.inf) / psf_flux if psf_flux > 0.0 else np.clip(psf, 0.0, np.inf)
        psf_radius, psf_profile, _psf_error = _radial_surface_brightness_profile(
            psf_unit,
            band.pixel_scale,
            zeropoint=0.0,
            center_xy=((psf.shape[1] - 1) / 2.0, (psf.shape[0] - 1) / 2.0),
            bin_width_arcsec=max(float(band.pixel_scale), 1.0e-12),
        )
        _plot_profile(psf_profile_ax, psf_radius, psf_profile, "-", lw=1.8, color="black")
        psf_profile_ax.invert_yaxis()
        psf_profile_ax.set_title(f"PSF profile: {band.filter_name}")
        psf_profile_ax.set_xlabel("Radius (arcsec)")
        psf_profile_ax.set_ylabel(r"$\mu_{\rm rel}$ (mag arcsec$^{-2}$)", fontsize=10)
        psf_profile_ax.yaxis.set_label_position("right")
        psf_profile_ax.yaxis.tick_right()
    return fig, axes


def plot_detection(*args, **kwargs):
    """Backward-compatible alias for plot_config."""

    return plot_config(*args, **kwargs)


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
