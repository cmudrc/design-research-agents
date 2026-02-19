"""Workflow facade exports with lazy pattern loading."""

from __future__ import annotations

from importlib import import_module
from typing import Final

from .workflow import Workflow

_EXPORTS: Final[dict[str, str]] = {
    "BlackboardPattern": "design_research_agents.implementations.patterns:BlackboardPattern",
    "ConversationPattern": "design_research_agents.implementations.patterns:ConversationPattern",
    "DebatePattern": "design_research_agents.implementations.patterns:DebatePattern",
    "NetworkedPattern": "design_research_agents.implementations.patterns:NetworkedPattern",
    "PlannerExecutorPattern": (
        "design_research_agents.implementations.patterns:PlannerExecutorPattern"
    ),
    "RagReasoningPattern": "design_research_agents.implementations.patterns:RagReasoningPattern",
    "ReflexionPattern": "design_research_agents.implementations.patterns:ReflexionPattern",
    "RouterPattern": "design_research_agents.implementations.patterns:RouterPattern",
    "TreeSearchPattern": "design_research_agents.implementations.patterns:TreeSearchPattern",
}

__all__ = ["Workflow", *_EXPORTS.keys()]


def __getattr__(name: str) -> object:
    """Lazily resolve exported workflow symbols.

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
    """Return workflow module attributes including lazy exports.

    Returns:
        Sorted attribute names visible on this module.
    """
    return sorted(set(globals()) | set(__all__))
