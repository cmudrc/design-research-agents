"""Reusable configurable workflow facade for user-supplied step graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal
from uuid import uuid4

from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import (
    WorkflowDelegate,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowResult,
    WorkflowStep,
)
from design_research_agents.schemas import validate_payload_against_schema
from design_research_agents.tracing import Tracer

from .workflow_runtime import WorkflowRuntime

WorkflowInputMode = Literal["prompt", "schema"]


def _normalize_steps(steps: Sequence[WorkflowStep]) -> tuple[WorkflowStep, ...]:
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


def _normalize_prompt(input_data: object) -> str:
    if not isinstance(input_data, str):
        raise ValueError("Workflow configured with input_mode='prompt' requires string input.")
    normalized_prompt = input_data.strip()
    if not normalized_prompt:
        raise ValueError("Workflow prompt input must be a non-empty string.")
    return normalized_prompt


def _normalize_inputs(input_data: object) -> dict[str, object]:
    if input_data is None:
        return {}
    if not isinstance(input_data, Mapping):
        raise ValueError("Workflow configured with input_mode='schema' requires mapping input.")
    return dict(input_data)


class Workflow:
    """Configured workflow for user-defined step graphs and run defaults."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime,
        steps: Sequence[WorkflowStep],
        agents: Mapping[str, WorkflowDelegate] | None = None,
        input_mode: WorkflowInputMode = "prompt",
        input_schema: Mapping[str, object] | None = None,
        prompt_context_key: str = "prompt",
        base_context: Mapping[str, object] | None = None,
        default_execution_mode: WorkflowExecutionMode = "sequential",
        default_failure_policy: WorkflowFailurePolicy = "skip_dependents",
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store runtime dependencies, step graph, and input handling mode."""
        normalized_mode: WorkflowInputMode = input_mode
        if normalized_mode not in {"prompt", "schema"}:
            raise ValueError("input_mode must be either 'prompt' or 'schema'.")
        normalized_prompt_context_key = prompt_context_key.strip()
        if normalized_mode == "prompt" and not normalized_prompt_context_key:
            raise ValueError("prompt_context_key must be non-empty for input_mode='prompt'.")

        self._runtime = WorkflowRuntime(
            tool_runtime=tool_runtime,
            agents=agents,
            tracer=tracer,
        )
        self._steps = _normalize_steps(steps)
        self._input_mode = normalized_mode
        self._input_schema = input_schema
        self._prompt_context_key = normalized_prompt_context_key or "prompt"
        self._base_context = dict(base_context or {})
        self._default_execution_mode = default_execution_mode
        self._default_failure_policy = default_failure_policy
        self._default_request_id_prefix = _normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})

    def run(
        self,
        input_data: str | Mapping[str, object] | None = None,
        *,
        execution_mode: WorkflowExecutionMode | None = None,
        failure_policy: WorkflowFailurePolicy | None = None,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute one workflow run with input interpreted by ``input_mode``."""
        resolved_request_id = _resolve_request_id(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        context = dict(self._base_context)
        if self._input_mode == "prompt":
            normalized_prompt = _normalize_prompt(input_data)
            context[self._prompt_context_key] = normalized_prompt
        else:
            normalized_inputs = _normalize_inputs(input_data)
            validate_payload_against_schema(
                payload=normalized_inputs,
                schema=self._input_schema,
                location="inputs",
            )
            context["inputs"] = normalized_inputs

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
    "Workflow",
    "WorkflowInputMode",
]
