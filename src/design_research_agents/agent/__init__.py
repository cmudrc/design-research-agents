"""Public agent facade exports with lazy loading."""

from __future__ import annotations

from importlib import import_module
from typing import Final

_EXPORTS: Final[dict[str, str]] = {
    "DirectLLMCall": "design_research_agents.implementations.agents:DirectLLMCall",
    "MultiStepCodeToolCallingAgent": (
        "design_research_agents.implementations.agents:MultiStepCodeToolCallingAgent"
    ),
    "MultiStepDirectLLMAgent": (
        "design_research_agents.implementations.agents:MultiStepDirectLLMAgent"
    ),
    "MultiStepJsonToolCallingAgent": (
        "design_research_agents.implementations.agents:MultiStepJsonToolCallingAgent"
    ),
    "MultiStepToolRouterAgent": (
        "design_research_agents.implementations.agents:MultiStepToolRouterAgent"
    ),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> object:
    """Lazily resolve exported agent symbols.

    Args:
        name: Exported symbol name requested by the caller.

    Returns:
        Resolved exported symbol object.

    Raises:
        AttributeError: Raised when ``name`` is not part of the public exports.
    """
    export_ref = _EXPORTS.get(name)
    if export_ref is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = export_ref.split(":")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes including lazy exports.

    Returns:
        Sorted attribute names visible on this module.
    """
    return sorted(set(globals()) | set(__all__))
