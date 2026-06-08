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
from .fastphot import BatchMatchedFilterResult, MatchedFilterResult, matched_filter_flux, matched_filter_flux_batch, moffat_alpha_from_fwhm, moffat_unit_flux, moffat_unit_flux_batch
from .grahspj import build_grahspj_config_from_image_bands, image_band_photometry_mjy
from .initialization import (
    estimate_scene_fluxes_from_pixels,
    estimate_sed_fluxes_from_pixels,
    initialize_sed_component_amplitudes_from_pixels,
)
from .io import download_galight_hsc_example, load_hsc_band
from .io import (
    combine_empirical_psf_results,
    construct_empirical_psf,
    build_empirical_psfs_for_bands,
    build_empirical_psfs_for_galex_bands,
    download_galex_coadd_blocks,
    download_legacy_survey_bricks_table,
    download_legacy_survey_coadd_band,
    EmpiricalPsfBandResult,
    EmpiricalPsfConfig,
    EmpiricalPsfResult,
    find_legacy_survey_brick,
    find_empirical_psf_candidates,
    GalexCoaddProduct,
    legacy_survey_bricks_url,
    legacy_survey_coadd_url,
    load_galex_coadd_band,
    load_legacy_survey_coadd_band,
    PsfCandidate,
    query_galex_coadd_products,
    read_legacy_survey_coadd_image,
)
from .model import jaguar_model, render_joint_model
from .mplstyle import style_path, use_style
from .plotting import plot_config, plot_detection, plot_empirical_psf_selection, plot_psf_candidates
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
    "EmpiricalPsfBandResult",
    "EmpiricalPsfConfig",
    "EmpiricalPsfResult",
    "GalexCoaddProduct",
    "PsfCandidate",
    "SourceDetectionConfig",
    "JaguarResult",
    "SedPoint",
    "fit",
    "fit_map",
    "BatchMatchedFilterResult",
    "MatchedFilterResult",
    "matched_filter_flux",
    "matched_filter_flux_batch",
    "moffat_alpha_from_fwhm",
    "moffat_unit_flux",
    "moffat_unit_flux_batch",
    "combine_empirical_psf_results",
    "build_grahspj_config_from_image_bands",
    "build_components_from_detections",
    "build_detection_image",
    "detect_sources",
    "image_band_photometry_mjy",
    "estimate_scene_fluxes_from_pixels",
    "estimate_sed_fluxes_from_pixels",
    "initialize_sed_component_amplitudes_from_pixels",
    "jaguar_model",
    "download_galex_coadd_blocks",
    "download_galight_hsc_example",
    "download_legacy_survey_bricks_table",
    "download_legacy_survey_coadd_band",
    "find_legacy_survey_brick",
    "legacy_survey_bricks_url",
    "legacy_survey_coadd_url",
    "build_empirical_psfs_for_bands",
    "build_empirical_psfs_for_galex_bands",
    "construct_empirical_psf",
    "find_empirical_psf_candidates",
    "query_galex_coadd_products",
    "read_legacy_survey_coadd_image",
    "load_hsc_band",
    "load_galex_coadd_band",
    "load_legacy_survey_coadd_band",
    "render_joint_model",
    "style_path",
    "use_style",
    "plot_config",
    "plot_detection",
    "plot_empirical_psf_selection",
    "plot_psf_candidates",
]
