"""JAGUAR: Joint AGN-Galaxy Unified Analysis & Reconstruction."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .config import (
    ComponentFluxes,
    ImageBandData,
    ImageFitConfig,
    JointFitConfig,
    MorphologyPriors,
    SceneComponentConfig,
    SedComponentConfig,
)
from .detection import (
    DetectedSource,
    SourceDetectionConfig,
    build_components_from_detections,
    build_detection_image,
    detect_sources,
)
from .fit import fit, fit_map
from .grahspj import build_grahspj_config_from_image_bands, image_band_photometry_mjy
from .initialization import (
    estimate_scene_fluxes_from_pixels,
    estimate_sed_fluxes_from_pixels,
    initialize_sed_component_amplitudes_from_pixels,
)
from .io import download_galight_hsc_example, load_hsc_band
from .model import jaguar_model, render_joint_model
from .plotting import plot_detection
from .result import JaguarResult, SedPoint

__all__ = [
    "ComponentFluxes",
    "ImageBandData",
    "ImageFitConfig",
    "JointFitConfig",
    "MorphologyPriors",
    "SceneComponentConfig",
    "SedComponentConfig",
    "DetectedSource",
    "SourceDetectionConfig",
    "JaguarResult",
    "SedPoint",
    "fit",
    "fit_map",
    "build_grahspj_config_from_image_bands",
    "build_components_from_detections",
    "build_detection_image",
    "detect_sources",
    "image_band_photometry_mjy",
    "estimate_scene_fluxes_from_pixels",
    "estimate_sed_fluxes_from_pixels",
    "initialize_sed_component_amplitudes_from_pixels",
    "jaguar_model",
    "download_galight_hsc_example",
    "load_hsc_band",
    "render_joint_model",
    "plot_detection",
]
