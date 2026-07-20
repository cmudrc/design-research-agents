"""Shared implementation exports, resolved lazily to keep runtime layers independent."""

from __future__ import annotations

from typing import Final

from design_research_agents._lazy_exports import module_dir, resolve_lazy_export

_AGENT_MODULE = "design_research_agents._implementations._agents"
_PATTERN_MODULE = "design_research_agents._implementations._patterns"

_EXPORTS: Final[dict[str, str]] = {
    "AdaptiveSchedule": f"{_PATTERN_MODULE}:AdaptiveSchedule",
    "BlackboardPattern": f"{_PATTERN_MODULE}:BlackboardPattern",
    "DebatePattern": f"{_PATTERN_MODULE}:DebatePattern",
    "DirectLLMCall": f"{_AGENT_MODULE}:DirectLLMCall",
    "EpsilonGreedyPolicy": f"{_PATTERN_MODULE}:EpsilonGreedyPolicy",
    "ExponentialSchedule": f"{_PATTERN_MODULE}:ExponentialSchedule",
    "LinearSchedule": f"{_PATTERN_MODULE}:LinearSchedule",
    "LogarithmicSchedule": f"{_PATTERN_MODULE}:LogarithmicSchedule",
    "MultiStepAgent": f"{_AGENT_MODULE}:MultiStepAgent",
    "NominalTeamPattern": f"{_PATTERN_MODULE}:NominalTeamPattern",
    "PlanExecutePattern": f"{_PATTERN_MODULE}:PlanExecutePattern",
    "ProposeCriticPattern": f"{_PATTERN_MODULE}:ProposeCriticPattern",
    "RAGPattern": f"{_PATTERN_MODULE}:RAGPattern",
    "RalphLoopPattern": f"{_PATTERN_MODULE}:RalphLoopPattern",
    "ReinforcementLearningPattern": f"{_PATTERN_MODULE}:ReinforcementLearningPattern",
    "RoundBasedCoordinationPattern": f"{_PATTERN_MODULE}:RoundBasedCoordinationPattern",
    "RouterDelegatePattern": f"{_PATTERN_MODULE}:RouterDelegatePattern",
    "SimulatedAnnealingPattern": f"{_PATTERN_MODULE}:SimulatedAnnealingPattern",
    "TemperatureSchedule": f"{_PATTERN_MODULE}:TemperatureSchedule",
    "TreeSearchPattern": f"{_PATTERN_MODULE}:TreeSearchPattern",
    "TwoSpeakerConversationPattern": f"{_PATTERN_MODULE}:TwoSpeakerConversationPattern",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    """Resolve one implementation export on first access.

    Args:
        name: Exported implementation symbol requested by the caller.

    Returns:
        Resolved implementation object.

    Raises:
        AttributeError: If ``name`` is not an implementation export.
    """
    return resolve_lazy_export(
        module_name=__name__,
        exports=_EXPORTS,
        export_name=name,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    """Return module attributes, including deferred implementation exports.

    Returns:
        Sorted attribute names visible on this module.
    """
    return module_dir(globals(), __all__)
