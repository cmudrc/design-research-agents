"""Public reusable pattern exports with lazy loading."""

from __future__ import annotations

from importlib import import_module
from typing import Final

_EXPORTS: Final[dict[str, str]] = {
    "BlackboardPattern": "design_research_agents._implementations._patterns:BlackboardPattern",
    "ConversationPattern": "design_research_agents._implementations._patterns:ConversationPattern",
    "DebatePattern": "design_research_agents._implementations._patterns:DebatePattern",
    "NetworkedPattern": "design_research_agents._implementations._patterns:NetworkedPattern",
    "PlanExecutePattern": "design_research_agents._implementations._patterns:PlanExecutePattern",
    "RAGPattern": "design_research_agents._implementations._patterns:RAGPattern",
    "ReflexionPattern": "design_research_agents._implementations._patterns:ReflexionPattern",
    "AgentRoutingPattern": "design_research_agents._implementations._patterns:AgentRoutingPattern",
    "TreeSearchPattern": "design_research_agents._implementations._patterns:TreeSearchPattern",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> object:
    """Lazily resolve exported pattern symbols.

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
