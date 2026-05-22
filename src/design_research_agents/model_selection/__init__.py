"""Stable public facade for model catalogs, flights, hardware, and selection."""

from __future__ import annotations

from design_research_agents._model_selection import (
    ModelCatalog,
    ModelFlight,
    ModelFlightRegistry,
    ModelSelector,
)
from design_research_agents._model_selection._hardware import HardwareProfile
from design_research_agents._model_selection._types import (
    ModelCostHint,
    ModelLatencyHint,
    ModelMemoryHint,
    ModelSafetyConstraints,
    ModelSelectionConstraints,
    ModelSelectionDecision,
    ModelSelectionIntent,
    ModelSelectionPolicyConfig,
    ModelSpec,
)

__all__ = [
    "HardwareProfile",
    "ModelCatalog",
    "ModelCostHint",
    "ModelFlight",
    "ModelFlightRegistry",
    "ModelLatencyHint",
    "ModelMemoryHint",
    "ModelSafetyConstraints",
    "ModelSelectionConstraints",
    "ModelSelectionDecision",
    "ModelSelectionIntent",
    "ModelSelectionPolicyConfig",
    "ModelSelector",
    "ModelSpec",
]


def __dir__() -> list[str]:
    """Return model-selection facade attributes, including re-exported contracts.

    Returns:
        Sorted attribute names visible on this module.
    """
    return sorted(set(globals()) | set(__all__))
