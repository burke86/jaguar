from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import ComponentFluxes


@dataclass
class FixedFluxBridge:
    """Simple bridge useful for tests and staged grahspj integration."""

    fluxes: Mapping[str, ComponentFluxes | Mapping[str, float]]

    def __call__(self, _sampled: Mapping[str, Any], filter_names: Sequence[str]) -> dict[str, ComponentFluxes]:
        out: dict[str, ComponentFluxes] = {}
        for name in filter_names:
            value = self.fluxes[name]
            if isinstance(value, ComponentFluxes):
                out[name] = value
            else:
                out[name] = ComponentFluxes(**value)
        return out


@dataclass
class GrahspjFluxBridge:
    """Callable wrapper for a future grahspj differentiable component evaluator."""

    evaluator: Any
    context: Any

    def __call__(self, sampled: Mapping[str, Any], filter_names: Sequence[str]) -> dict[str, ComponentFluxes]:
        flux_table = self.evaluator(self.context, sampled)
        out = {}
        for name in filter_names:
            if name not in flux_table:
                raise KeyError(f"grahspj flux evaluator did not return image filter {name!r}.")
            out[name] = ComponentFluxes(**flux_table[name])
        return out

