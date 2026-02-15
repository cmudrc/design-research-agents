"""Reusable pure-tool workflow orchestration facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import (
    AgentStep,
    LogicStep,
    ToolStep,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowResult,
)
from design_research_agents.schemas import validate_payload_against_schema

from .workflow_runtime import WorkflowRuntime

PureWorkflowStep = LogicStep | ToolStep


def _normalize_steps(steps: Sequence[PureWorkflowStep]) -> tuple[PureWorkflowStep, ...]:
    if not steps:
        raise ValueError("'steps' must contain at least one workflow step.")
    for step in steps:
        if isinstance(step, AgentStep):
            raise ValueError(
                "Pure tool workflow does not accept AgentStep entries. "
                "Use LogicStep and ToolStep only."
            )
    return tuple(steps)


class PureToolWorkflow:
    """Configured pure-tool workflow for user-supplied step graphs."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime,
        steps: Sequence[PureWorkflowStep],
        input_schema: Mapping[str, object] | None = None,
        base_context: Mapping[str, object] | None = None,
    ) -> None:
        """Store runtime dependencies, step graph, and optional input schema."""
        self._runtime = WorkflowRuntime(tool_runtime=tool_runtime)
        self._steps = _normalize_steps(steps)
        self._input_schema = input_schema
        self._base_context = dict(base_context or {})

    def run(
        self,
        *,
        inputs: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode = "sequential",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute one pure-tool workflow run with optional run-scoped inputs."""
        resolved_inputs = dict(inputs or {})
        validate_payload_against_schema(
            payload=resolved_inputs,
            schema=self._input_schema,
            location="inputs",
        )
        context = dict(self._base_context)
        context["inputs"] = resolved_inputs
        return self._runtime.run(
            self._steps,
            context=context,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            request_id=request_id,
            dependencies=dependencies,
        )


def pure_tool_workflow(
    *,
    tool_runtime: ToolRuntime,
    steps: Sequence[PureWorkflowStep],
    input_schema: Mapping[str, object] | None = None,
    base_context: Mapping[str, object] | None = None,
) -> PureToolWorkflow:
    """Return a configured pure-tool workflow orchestration chunk."""
    return PureToolWorkflow(
        tool_runtime=tool_runtime,
        steps=steps,
        input_schema=input_schema,
        base_context=base_context,
    )


__all__ = [
    "PureToolWorkflow",
    "PureWorkflowStep",
    "pure_tool_workflow",
]
