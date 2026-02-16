"""Reusable pure-tool workflow orchestration facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import uuid4

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
from design_research_agents.tracing import Tracer

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


class PureToolWorkflow:
    """Configured pure-tool workflow for user-supplied step graphs."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime,
        steps: Sequence[PureWorkflowStep],
        input_schema: Mapping[str, object] | None = None,
        base_context: Mapping[str, object] | None = None,
        default_execution_mode: WorkflowExecutionMode = "sequential",
        default_failure_policy: WorkflowFailurePolicy = "skip_dependents",
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store runtime dependencies, step graph, and optional input schema.

        Args:
            tool_runtime: The ToolRuntime instance to use for executing tool steps in this
                workflow.
            steps: The sequence of LogicStep and ToolStep instances that define the step
                graph for this workflow. Must contain at least one step and cannot contain
                AgentStep entries.
            input_schema: Optional JSON Schema to validate run inputs against at runtime.
                If provided, the workflow will validate the inputs passed to each run against
                this schema and raise a ValueError if validation fails.
            base_context: Optional mapping of context values to include in the execution
                context for every run of this workflow. This can be used to provide static
                context values that are needed by the steps in the workflow.
            default_execution_mode: The default WorkflowExecutionMode to use for runs of
                this workflow when  not explicitly specified at run time. Defaults to
                "sequential".
            default_failure_policy: The default WorkflowFailurePolicy to use for runs
                of this workflow when not explicitly specified at run time. Defaults to
                "skip_dependents".
            default_request_id_prefix: Optional default prefix to use when generating
                request IDs for runs of this workflow that don't provide their own request
                ID. If provided, must be a non-empty string. This prefix can be used to
                make it easier to identify and group runs of this workflow `in logs and traces.
            default_dependencies: Optional mapping of default dependencies to provide for all
                runs of this workflow, which can be overridden by dependencies provided at
                run time. This can  be used to provide static dependencies that are needed
                by the steps in the workflow.
            tracer: Optional Tracer instance to use for emitting events during execution of this
                workflow.
        """
        self._runtime = WorkflowRuntime(tool_runtime=tool_runtime, tracer=tracer)
        self._steps = _normalize_steps(steps)
        self._input_schema = input_schema
        self._base_context = dict(base_context or {})
        self._default_execution_mode = default_execution_mode
        self._default_failure_policy = default_failure_policy
        self._default_request_id_prefix = _normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})

    def run(
        self,
        *,
        inputs: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode | None = None,
        failure_policy: WorkflowFailurePolicy | None = None,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute one pure-tool workflow run with optional run-scoped inputs.

        Args:
            inputs: Optional mapping of input values to provide for this run. If the workflow
                was configured with an input_schema, these inputs will be validated against
                that schema before execution.
            execution_mode: Optional WorkflowExecutionMode to use for this run. If not provided,
                the default execution mode configured for this workflow will be used.
            failure_policy: Optional WorkflowFailurePolicy to use for this run. If not provided,
                the default failure policy configured for this workflow will be used.
            request_id: Optional request ID to use for this run. If not provided, a request
                ID will be generated using the default prefix configured for this workflow (if
                any).
            dependencies: Optional mapping of dependencies to provide for this run,
                which will override any default dependencies configured for this workflow.
                This can be used to provide run-specific dependencies that are needed by the
                steps in the workflow.
        """
        resolved_inputs = dict(inputs or {})
        validate_payload_against_schema(
            payload=resolved_inputs,
            schema=self._input_schema,
            location="inputs",
        )
        resolved_request_id = _resolve_request_id(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        context = dict(self._base_context)
        context["inputs"] = resolved_inputs
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
    "PureToolWorkflow",
    "PureWorkflowStep",
]
