from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ComponentFluxes:
    """Image-band component fluxes in image count units."""

    agn: float
    host: float
    star: float = 0.0
    dust: float = 0.0
    nebular: float = 0.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"Component flux {name!r} must be finite.")
            if float(value) < 0.0:
                raise ValueError(f"Component flux {name!r} must be non-negative.")


@dataclass(frozen=True)
class ImageBandData:
    """One aligned image band used by the joint pixel likelihood."""

    image: np.ndarray
    noise: np.ndarray
    psf: np.ndarray
    filter_name: str
    pixel_scale: float
    psf_uncertainty: np.ndarray | None = None
    effective_wavelength: float | None = None
    zeropoint: float | None = None
    counts_per_mjy: float | None = None
    mask: np.ndarray | None = None
    header: Mapping[str, Any] | None = None
    target_pixel: tuple[float, float] | None = None
    psf_padding_pixels: int = 0

    def validate(self) -> None:
        if self.image.ndim != 2:
            raise ValueError("image must be a 2D array.")
        if self.noise.shape != self.image.shape:
            raise ValueError("noise must match image shape.")
        if self.psf.ndim != 2:
            raise ValueError("psf must be a 2D array.")
        if self.psf_uncertainty is not None and np.asarray(self.psf_uncertainty).shape != self.psf.shape:
            raise ValueError("psf_uncertainty must match psf shape.")
        if self.mask is not None and self.mask.shape != self.image.shape:
            raise ValueError("mask must match image shape.")
        if not np.isfinite(self.pixel_scale) or self.pixel_scale <= 0.0:
            raise ValueError("pixel_scale must be positive and finite.")
        if self.counts_per_mjy is not None and (not np.isfinite(self.counts_per_mjy) or self.counts_per_mjy <= 0.0):
            raise ValueError("counts_per_mjy must be positive and finite when set.")
        if int(self.psf_padding_pixels) != self.psf_padding_pixels or self.psf_padding_pixels < 0:
            raise ValueError("psf_padding_pixels must be a non-negative integer.")
        if np.any(~np.isfinite(self.image)):
            raise ValueError("image contains non-finite values.")
        if np.any(~np.isfinite(self.noise)) or np.any(self.noise <= 0.0):
            raise ValueError("noise must be finite and strictly positive.")
        if np.any(~np.isfinite(self.psf)):
            raise ValueError("psf contains non-finite values.")
        if self.psf_uncertainty is not None:
            psf_uncertainty = np.asarray(self.psf_uncertainty)
            if np.any(~np.isfinite(psf_uncertainty)) or np.any(psf_uncertainty < 0.0):
                raise ValueError("psf_uncertainty must be finite and non-negative.")


@dataclass(frozen=True)
class MorphologyPriors:
    """Host and point-source morphology priors."""

    center_sigma_pix: float = 1.0
    reff_arcsec_loc: float = 0.5
    reff_arcsec_sigma: float = 0.5
    n_sersic_loc: float = 2.0
    n_sersic_sigma: float = 1.0
    ellipticity_sigma: float = 0.25
    background_sigma: float = 10.0


@dataclass(frozen=True)
class ImageFitConfig:
    """Image-likelihood options."""

    fit_background: bool = True
    background_default: float = 0.0
    fit_host_morphology: bool = True
    fixed_reff_arcsec: float = 0.5
    fixed_n_sersic: float = 2.0
    fixed_e1: float = 0.0
    fixed_e2: float = 0.0
    fixed_center_x_pix: float = 0.0
    fixed_center_y_pix: float = 0.0
    student_t_df: float = 5.0
    use_student_t: bool = True
    priors: MorphologyPriors = field(default_factory=MorphologyPriors)


FluxModel = Callable[[Mapping[str, Any], Sequence[str]], Mapping[str, ComponentFluxes]]


@dataclass
class SedComponentConfig:
    """One SED component used by the joint image and photometry model."""

    name: str
    kind: str
    spatial: str | None = None
    grahspj_config: Any | None = None
    grahspj_context: Any | None = None
    temperature_k: float = 5800.0
    fit_temperature: bool = False
    reference_filter_name: str | None = None
    reference_flux_mjy: float = 1.0
    fit_reference_flux: bool = True
    log_reference_flux_sigma: float = 3.0

    def validate(self) -> None:
        valid_kinds = {"agn", "host", "galaxy", "star"}
        if self.kind not in valid_kinds:
            raise ValueError(f"SED component {self.name!r} kind must be one of {sorted(valid_kinds)}.")
        valid_spatial = {"point", "extended"}
        spatial = self.resolved_spatial
        if spatial not in valid_spatial:
            raise ValueError(f"SED component {self.name!r} spatial role must be one of {sorted(valid_spatial)}.")
        if self.kind in {"agn", "host", "galaxy"} and self.grahspj_config is None:
            raise ValueError(f"SED component {self.name!r} requires grahspj_config.")
        if not np.isfinite(float(self.temperature_k)) or float(self.temperature_k) <= 0.0:
            raise ValueError("star temperature_k must be positive and finite.")
        if not np.isfinite(float(self.reference_flux_mjy)) or float(self.reference_flux_mjy) <= 0.0:
            raise ValueError("star reference_flux_mjy must be positive and finite.")
        if not np.isfinite(float(self.log_reference_flux_sigma)) or float(self.log_reference_flux_sigma) <= 0.0:
            raise ValueError("star log_reference_flux_sigma must be positive and finite.")

    @property
    def resolved_spatial(self) -> str:
        if self.spatial is not None:
            return self.spatial
        return "point" if self.kind in {"agn", "star"} else "extended"


@dataclass
class SceneComponentConfig:
    """One spatial component in the image scene."""

    name: str
    sed_component: str
    kind: str
    fit_position: bool = True
    fixed_center_x_pix: float = 0.0
    fixed_center_y_pix: float = 0.0
    center_sigma_pix: float | None = None
    fit_shape: bool = True
    fixed_reff_arcsec: float = 0.5
    fixed_n_sersic: float = 2.0
    fixed_e1: float = 0.0
    fixed_e2: float = 0.0
    reff_arcsec_loc: float | None = None
    reff_arcsec_sigma: float | None = None
    min_reff_arcsec: float = 1.0e-3
    max_reff_arcsec: float | None = None
    n_sersic_loc: float | None = None
    n_sersic_sigma: float | None = None
    ellipticity_sigma: float | None = None

    def validate(self) -> None:
        valid_kinds = {"point", "sersic"}
        if self.kind not in valid_kinds:
            raise ValueError(f"Scene component {self.name!r} kind must be one of {sorted(valid_kinds)}.")
        for attr in ("fixed_center_x_pix", "fixed_center_y_pix"):
            if not np.isfinite(float(getattr(self, attr))):
                raise ValueError(f"Scene component {self.name!r} {attr} must be finite.")
        if self.center_sigma_pix is not None and (not np.isfinite(float(self.center_sigma_pix)) or float(self.center_sigma_pix) <= 0.0):
            raise ValueError(f"Scene component {self.name!r} center_sigma_pix must be positive and finite.")
        if self.kind == "sersic":
            if not np.isfinite(float(self.fixed_reff_arcsec)) or float(self.fixed_reff_arcsec) <= 0.0:
                raise ValueError(f"Scene component {self.name!r} fixed_reff_arcsec must be positive and finite.")
            if not np.isfinite(float(self.fixed_n_sersic)) or float(self.fixed_n_sersic) <= 0.0:
                raise ValueError(f"Scene component {self.name!r} fixed_n_sersic must be positive and finite.")
            if not np.isfinite(float(self.min_reff_arcsec)) or float(self.min_reff_arcsec) <= 0.0:
                raise ValueError(f"Scene component {self.name!r} min_reff_arcsec must be positive and finite.")
            if self.max_reff_arcsec is not None:
                if not np.isfinite(float(self.max_reff_arcsec)) or float(self.max_reff_arcsec) <= float(self.min_reff_arcsec):
                    raise ValueError(f"Scene component {self.name!r} max_reff_arcsec must be finite and larger than min_reff_arcsec.")


@dataclass
class JointFitConfig:
    """Full joint image + SED fitting configuration."""

    image_bands: Sequence[ImageBandData]
    image: ImageFitConfig = field(default_factory=ImageFitConfig)
    grahspj_config: Any | None = None
    grahspj_context: Any | None = None
    sed_components: Sequence[SedComponentConfig] | None = None
    scene_components: Sequence[SceneComponentConfig] | None = None
    fixed_component_fluxes: Mapping[str, ComponentFluxes | Mapping[str, float]] | None = None
    component_flux_model: FluxModel | None = None
    joint_grahspj_fitting: bool = False
    image_flux_prior_sigma: float = 3.0

    def validate(self) -> None:
        if not self.image_bands:
            raise ValueError("At least one image band is required.")
        if self.grahspj_config is None and not self.sed_components:
            raise ValueError("grahspj_config or sed_components is required for unified JAGUAR fitting.")
        if not np.isfinite(float(self.image_flux_prior_sigma)) or float(self.image_flux_prior_sigma) <= 0.0:
            raise ValueError("image_flux_prior_sigma must be positive and finite.")
        component_names: set[str] = set()
        for component in self.resolved_sed_components:
            component.validate()
            if component.name in component_names:
                raise ValueError(f"Duplicate SED component name {component.name!r}.")
            component_names.add(component.name)
        scene_names: set[str] = set()
        for component in self.resolved_scene_components:
            component.validate()
            if component.name in scene_names:
                raise ValueError(f"Duplicate scene component name {component.name!r}.")
            if component.sed_component not in component_names:
                raise ValueError(f"Scene component {component.name!r} references unknown SED component {component.sed_component!r}.")
            scene_names.add(component.name)
        seen: set[str] = set()
        for band in self.image_bands:
            band.validate()
            if band.filter_name in seen:
                raise ValueError(f"Duplicate image filter name {band.filter_name!r}.")
            seen.add(band.filter_name)

    @property
    def resolved_sed_components(self) -> Sequence[SedComponentConfig]:
        if self.sed_components:
            return self.sed_components
        if self.grahspj_config is None:
            return ()
        self.sed_components = (
            SedComponentConfig(name="agn", kind="agn", spatial="point", grahspj_config=self.grahspj_config),
            SedComponentConfig(name="host", kind="host", spatial="extended", grahspj_config=self.grahspj_config),
        )
        return self.sed_components

    @property
    def resolved_scene_components(self) -> Sequence[SceneComponentConfig]:
        if self.scene_components:
            return self.scene_components
        components = []
        for component in self.resolved_sed_components:
            kind = "point" if component.resolved_spatial == "point" else "sersic"
            components.append(
                SceneComponentConfig(
                    name=component.name,
                    sed_component=component.name,
                    kind=kind,
                    fit_position=self.image.fit_host_morphology,
                    fixed_center_x_pix=self.image.fixed_center_x_pix,
                    fixed_center_y_pix=self.image.fixed_center_y_pix,
                    fit_shape=self.image.fit_host_morphology,
                    fixed_reff_arcsec=self.image.fixed_reff_arcsec,
                    fixed_n_sersic=self.image.fixed_n_sersic,
                    fixed_e1=self.image.fixed_e1,
                    fixed_e2=self.image.fixed_e2,
                )
            )
        self.scene_components = tuple(components)
        return self.scene_components


def coerce_component_fluxes(value: ComponentFluxes | Mapping[str, float]) -> ComponentFluxes:
    """Convert mapping-like flux payloads into ComponentFluxes."""

    if isinstance(value, ComponentFluxes):
        fluxes = value
    else:
        fluxes = ComponentFluxes(
            agn=float(value.get("agn", 0.0)),
            host=float(value.get("host", 0.0)),
            star=float(value.get("star", 0.0)),
            dust=float(value.get("dust", 0.0)),
            nebular=float(value.get("nebular", 0.0)),
        )
    fluxes.validate()
    return fluxes
