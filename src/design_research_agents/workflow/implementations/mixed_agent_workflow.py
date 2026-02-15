"""Reusable mixed workflow orchestration facade (logic + agent + tools)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import (
    AgentStep,
    LogicStep,
    ToolStep,
    WorkflowDelegate,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowResult,
)

from .workflow_runtime import WorkflowRuntime

MixedWorkflowStep = LogicStep | AgentStep | ToolStep


def _normalize_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("prompt must be a non-empty string.")
    return normalized_prompt


def _normalize_agents(agents: Mapping[str, WorkflowDelegate]) -> dict[str, WorkflowDelegate]:
    normalized_agents = {
        name.strip(): delegate
        for name, delegate in agents.items()
        if isinstance(name, str) and name.strip()
    }
    if not normalized_agents:
        raise ValueError("'agents' must include at least one non-empty key.")
    return normalized_agents


def _normalize_steps(steps: Sequence[MixedWorkflowStep]) -> tuple[MixedWorkflowStep, ...]:
    if not steps:
        raise ValueError("'steps' must contain at least one workflow step.")
    return tuple(steps)


class MixedAgentWorkflow:
    """Configured mixed workflow for user-supplied step graphs."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime,
        agents: Mapping[str, WorkflowDelegate],
        steps: Sequence[MixedWorkflowStep],
        base_context: Mapping[str, object] | None = None,
    ) -> None:
        """Store runtime dependencies, delegate bindings, and step topology."""
        self._runtime = WorkflowRuntime(
            tool_runtime=tool_runtime,
            agents=_normalize_agents(agents),
        )
        self._steps = _normalize_steps(steps)
        self._base_context = dict(base_context or {})

    def run(
        self,
        prompt: str,
        *,
        execution_mode: WorkflowExecutionMode = "dag",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute one mixed workflow run with a required prompt."""
        normalized_prompt = _normalize_prompt(prompt)
        context = dict(self._base_context)
        context["prompt"] = normalized_prompt
        return self._runtime.run(
            self._steps,
            context=context,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            request_id=request_id,
            dependencies=dependencies,
        )


def mixed_agent_workflow(
    *,
    tool_runtime: ToolRuntime,
    agents: Mapping[str, WorkflowDelegate],
    steps: Sequence[MixedWorkflowStep],
    base_context: Mapping[str, object] | None = None,
) -> MixedAgentWorkflow:
    """Return a configured mixed workflow orchestration chunk."""
    return MixedAgentWorkflow(
        tool_runtime=tool_runtime,
        agents=agents,
        steps=steps,
        base_context=base_context,
    )


__all__ = [
    "MixedAgentWorkflow",
    "MixedWorkflowStep",
    "mixed_agent_workflow",
]
