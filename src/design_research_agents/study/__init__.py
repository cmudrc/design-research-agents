"""Stable public facade for study-oriented agent execution."""

from __future__ import annotations

from design_research_agents.integration import (
    AgentBinding,
    AgentExecutionEnvelope,
    AgentRunRequest,
    StudyCondition,
    execute_agent_request,
    execute_agent_run,
    normalize_agent_execution,
)

__all__ = [
    "AgentBinding",
    "AgentExecutionEnvelope",
    "AgentRunRequest",
    "StudyCondition",
    "execute_agent_request",
    "execute_agent_run",
    "normalize_agent_execution",
]


def __dir__() -> list[str]:
    """Return study facade attributes, including re-exported contracts.

    Returns:
        Sorted attribute names visible on this module.
    """
    return sorted(set(globals()) | set(__all__))
