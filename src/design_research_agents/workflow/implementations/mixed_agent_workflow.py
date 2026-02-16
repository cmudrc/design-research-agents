"""Reusable mixed workflow orchestration facade (logic + agent + tools)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import uuid4

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
from design_research_agents.tracing import Tracer

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


def _normalize_request_id_prefix(default_request_id_prefix: str | None) -> str | None:
    if default_request_id_prefix is None:
        return None
    normalized_prefix = default_request_id_prefix.strip()
    if not normalized_prefix:
        raise ValueError("default_request_id_prefix must be a non-empty string when provided.")
    return normalized_prefix


def _resolve_request_id(*, request_id: str | None, default_prefix: str | None) -> str | None:
    if request_id is not None and request_id.strip():
        return request_id
    if default_prefix is None:
        return request_id
    return f"{default_prefix}:{uuid4().hex}"


def _merge_dependencies(
    *,
    default_dependencies: Mapping[str, object],
    run_dependencies: Mapping[str, object] | None,
) -> dict[str, object]:
    merged = dict(default_dependencies)
    if run_dependencies is not None:
        merged.update(run_dependencies)
    return merged


class MixedAgentWorkflow:
    """Configured mixed workflow for user-supplied step graphs."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime,
        agents: Mapping[str, WorkflowDelegate],
        steps: Sequence[MixedWorkflowStep],
        base_context: Mapping[str, object] | None = None,
        default_execution_mode: WorkflowExecutionMode = "dag",
        default_failure_policy: WorkflowFailurePolicy = "skip_dependents",
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store runtime dependencies, delegate bindings, and step topology."""
        self._runtime = WorkflowRuntime(
            tool_runtime=tool_runtime,
            agents=_normalize_agents(agents),
            tracer=tracer,
        )
        self._steps = _normalize_steps(steps)
        self._base_context = dict(base_context or {})
        self._default_execution_mode = default_execution_mode
        self._default_failure_policy = default_failure_policy
        self._default_request_id_prefix = _normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})

    def run(
        self,
        prompt: str,
        *,
        execution_mode: WorkflowExecutionMode | None = None,
        failure_policy: WorkflowFailurePolicy | None = None,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute one mixed workflow run with a required prompt."""
        normalized_prompt = _normalize_prompt(prompt)
        context = dict(self._base_context)
        context["prompt"] = normalized_prompt
        resolved_request_id = _resolve_request_id(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        return self._runtime.run(
            self._steps,
            context=context,
            execution_mode=execution_mode or self._default_execution_mode,
            failure_policy=failure_policy or self._default_failure_policy,
            request_id=resolved_request_id,
            dependencies=_merge_dependencies(
                default_dependencies=self._default_dependencies,
                run_dependencies=dependencies,
            ),
        )


__all__ = [
    "MixedAgentWorkflow",
    "MixedWorkflowStep",
]
