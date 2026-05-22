"""Stable facade exports for model selection."""

from __future__ import annotations

from ._catalog import ModelCatalog, ModelFlight, ModelFlightRegistry
from ._selector import ModelSelector

__all__ = ["ModelCatalog", "ModelFlight", "ModelFlightRegistry", "ModelSelector"]
