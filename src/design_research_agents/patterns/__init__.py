"""Public reusable pattern exports with lazy loading."""

from __future__ import annotations

from typing import Final

from design_research_agents._lazy_exports import module_dir, resolve_lazy_export

_EXPORTS: Final[dict[str, str]] = {
    "BlackboardPattern": "design_research_agents._implementations._patterns:BlackboardPattern",
    "TwoSpeakerConversationPattern": "design_research_agents._implementations._patterns:TwoSpeakerConversationPattern",
    "DebatePattern": "design_research_agents._implementations._patterns:DebatePattern",
    "RoundBasedCoordinationPattern": "design_research_agents._implementations._patterns:RoundBasedCoordinationPattern",
    "PlanExecutePattern": "design_research_agents._implementations._patterns:PlanExecutePattern",
    "RAGPattern": "design_research_agents._implementations._patterns:RAGPattern",
    "ProposeCriticPattern": "design_research_agents._implementations._patterns:ProposeCriticPattern",
    "RouterDelegatePattern": "design_research_agents._implementations._patterns:RouterDelegatePattern",
    "BeamSearchPattern": "design_research_agents._implementations._patterns:BeamSearchPattern",
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
    return resolve_lazy_export(
        module_name=__name__,
        exports=_EXPORTS,
        export_name=name,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    """Return module attributes including lazy exports.

    Returns:
        Sorted attribute names visible on this module.
    """
    return module_dir(globals(), __all__)
