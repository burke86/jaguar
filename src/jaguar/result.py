from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ComponentFluxes, coerce_component_fluxes


@dataclass
class SedPoint:
    """One image-band SED component point."""

    filter_name: str
    wavelength: float
    agn: float
    host: float
    total: float
    dust: float = 0.0
    nebular: float = 0.0
    observed: float | None = None
    error: float | None = None


@dataclass
class JaguarResult:
    """Container for JAGUAR fit outputs."""

    config: Any
    map_params: dict[str, Any] | None
    samples: dict[str, Any] | None
    rendered: dict[str, dict[str, Any]]
    grahspj_state: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        params = self.map_params or {}
        for key, value in params.items():
            arr = np.asarray(value)
            if arr.size == 1:
                out[key] = float(arr)
        for name, components in self.rendered.items():
            model = np.asarray(components["total"], dtype=float)
            data = np.asarray(next(b.image for b in self.config.image_bands if b.filter_name == name), dtype=float)
            noise = np.asarray(next(b.noise for b in self.config.image_bands if b.filter_name == name), dtype=float)
            out[f"{name}_reduced_chi2"] = float(np.nanmean(((data - model) / noise) ** 2))
            out[f"{name}_model_flux"] = float(np.sum(model))
            out[f"{name}_agn_flux"] = float(np.sum(np.asarray(components["agn"])))
            if "star" in components:
                out[f"{name}_star_flux"] = float(np.sum(np.asarray(components["star"])))
            out[f"{name}_host_flux"] = float(np.sum(np.asarray(components["host"])))
        return out

    def sed_points(self) -> list[SedPoint]:
        """Return per-band SED points from rendered image component fluxes."""

        points: list[SedPoint] = []
        state = self.grahspj_state or {}
        pred_fluxes = np.asarray(state.get("pred_fluxes", []), dtype=float)
        agn_fluxes = np.asarray(state.get("agn_fluxes", []), dtype=float)
        host_fluxes = np.asarray(state.get("host_fluxes", []), dtype=float)
        star_fluxes = np.asarray(state.get("star_fluxes", np.zeros_like(pred_fluxes)), dtype=float)
        dust_fluxes = np.asarray(state.get("dust_fluxes", np.zeros_like(pred_fluxes)), dtype=float)
        nebular_fluxes = np.asarray(state.get("nebular_fluxes", np.zeros_like(pred_fluxes)), dtype=float)
        grahspj_wavelengths = np.asarray([], dtype=float)
        if self.config.grahspj_context is not None:
            grahspj_wavelengths = np.asarray(self.config.grahspj_context.filter_effective_wavelength_jax, dtype=float)
        elif self.config.grahspj_config is not None and getattr(self.config.grahspj_config, "photometry", None) is not None:
            # Context-free fallback for tests or partially constructed results.
            filters = getattr(self.config.grahspj_config, "filters", None)
            grahspj_wavelengths = np.asarray(
                [getattr(curve, "effective_wavelength", np.nan) for curve in getattr(filters, "curves", [])],
                dtype=float,
            )
        for index, band in enumerate(self.config.image_bands):
            name = band.filter_name
            if pred_fluxes.size:
                agn = float(agn_fluxes[index])
                star = float(star_fluxes[index]) if star_fluxes.size else 0.0
                host = float(host_fluxes[index])
                total = float(pred_fluxes[index])
                dust = float(dust_fluxes[index]) if dust_fluxes.size else 0.0
                nebular = float(nebular_fluxes[index]) if nebular_fluxes.size else 0.0
            else:
                components = self.rendered[name]
                agn = float(np.sum(np.asarray(components["agn"], dtype=float)))
                star = float(np.sum(np.asarray(components.get("star", 0.0), dtype=float)))
                host = float(np.sum(np.asarray(components["host"], dtype=float)))
                total = float(np.sum(np.asarray(components["total"], dtype=float)))
                dust = 0.0
                nebular = 0.0
            if index < grahspj_wavelengths.size and np.isfinite(grahspj_wavelengths[index]):
                wavelength = float(grahspj_wavelengths[index])
            elif band.effective_wavelength is not None:
                wavelength = float(band.effective_wavelength)
            else:
                wavelength = float(index + 1)
            if band.counts_per_mjy is not None:
                observed = float(np.sum(np.asarray(band.image, dtype=float)) / band.counts_per_mjy)
                error = float(np.sqrt(np.sum(np.asarray(band.noise, dtype=float) ** 2)) / band.counts_per_mjy)
            else:
                observed = float(np.sum(np.asarray(band.image, dtype=float)))
                error = float(np.sqrt(np.sum(np.asarray(band.noise, dtype=float) ** 2)))
            points.append(SedPoint(name, wavelength, agn + star, host, total, dust, nebular, observed, error))
        return sorted(points, key=lambda point: point.wavelength)

    def configured_component_fluxes(self) -> dict[str, ComponentFluxes]:
        """Return configured component fluxes when fixed fluxes were used."""

        if self.config.fixed_component_fluxes is None:
            return {}
        return {
            name: coerce_component_fluxes(value)
            for name, value in self.config.fixed_component_fluxes.items()
        }
