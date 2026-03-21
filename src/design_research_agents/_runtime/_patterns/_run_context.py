"""Shared helpers for workflow-native orchestration patterns."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from string import Template
from typing import Any, cast

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import ToolResult, ToolSpec
from design_research_agents._contracts._workflow import (
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
)
from design_research_agents._implementations._shared._agent_internal._prompt_overrides import (
    validate_prompt_text,
)
from design_research_agents._implementations._shared._agent_internal._result_builders import (
    build_failure_result,
)
from design_research_agents._implementations._shared._agent_internal._run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents._runtime._common._run_defaults import (
    merge_dependencies,
    resolve_request_id_with_prefix,
)
from design_research_agents._tracing import Tracer, finish_trace_run, start_trace_run
from design_research_agents._tracing._result_metadata import (
    enrich_execution_result_trace_metadata,
)
from design_research_agents.workflow import CompiledExecution, Workflow


@dataclass(slots=True, kw_only=True)
class WorkflowBudgetTracker:
    """Observed-metrics accumulator for workflow-native orchestration patterns."""

    observed_latency_ms: int = 0
    """Accumulated model latency across observed calls."""
    observed_model_calls: int = 0
    """Count of observed model calls."""
    observed_tool_calls: int = 0
    """Count of observed tool calls."""
    observed_estimated_usd: float = 0.0
    """Accumulated estimated tool spend in USD."""

    def add_model_response(self, model_response: LLMResponse | None) -> None:
        """Accumulate model-call metrics from one optional response.

        Args:
            model_response: Model response whose metrics should be counted.
        """
        if model_response is None:
            return
        self.observed_model_calls += 1
        if isinstance(model_response.latency_ms, int) and model_response.latency_ms >= 0:
            self.observed_latency_ms += model_response.latency_ms

    def add_tool_results(
        self,
        *,
        tool_results: list[ToolResult],
        tool_specs: Mapping[str, ToolSpec],
    ) -> None:
        """Accumulate tool-call counts and estimated tool spend.

        Args:
            tool_results: Tool invocation results emitted by a workflow step.
            tool_specs: Tool specs indexed by tool name for cost lookup.
        """
        for tool_result in tool_results:
            self.observed_tool_calls += 1
            runtime_spec = tool_specs.get(tool_result.tool_name)
            if runtime_spec is None:
                continue
            estimated_cost = runtime_spec.cost_hints.usd_cost_estimate
            if isinstance(estimated_cost, (int, float)):
                self.observed_estimated_usd += float(estimated_cost)

    def as_metadata(self) -> dict[str, object]:
        """Return normalized observed-metrics metadata payload.

        Returns:
            Observed metrics payload suitable for runtime metadata.
        """
        return {
            "observed_latency_ms": self.observed_latency_ms,
            "observed_model_calls": self.observed_model_calls,
            "observed_tool_calls": self.observed_tool_calls,
            "observed_estimated_usd": round(self.observed_estimated_usd, 6),
        }


@dataclass(slots=True, frozen=True, kw_only=True)
class PatternRunContext:
    """Normalized request/dependency payload for one pattern run."""

    request_id: str
    """Resolved request id used for tracing and nested invocations."""
    dependencies: dict[str, object]
    """Merged and normalized run dependency mapping."""
    normalized_input: dict[str, object]
    """Normalized run input payload mapping."""
    prompt: str
    """Resolved prompt extracted from normalized input."""


def resolve_pattern_run_context(
    *,
    prompt: str | object,
    default_request_id_prefix: str | None,
    default_dependencies: Mapping[str, object],
    request_id: str | None,
    dependencies: Mapping[str, object] | None,
) -> PatternRunContext:
    """Resolve one pattern run context with shared request/dependency semantics."""
    # Prefix-based ids keep pattern families queryable in traces without requiring callers to pass explicit ids.
    configured_request_id = resolve_request_id_with_prefix(
        request_id=request_id,
        default_prefix=default_request_id_prefix,
    )
    resolved_request_id = resolve_request_id(configured_request_id)
    resolved_dependencies = normalize_dependencies(
        merge_dependencies(
            default_dependencies=default_dependencies,
            run_dependencies=dependencies,
        )
    )
    normalized_input = normalize_input_payload(prompt)
    return PatternRunContext(
        request_id=resolved_request_id,
        dependencies=resolved_dependencies,
        normalized_input=normalized_input,
        prompt=str(normalized_input.get("prompt", "")),
    )


def execute_pattern_with_trace(
    *,
    agent_name: str,
    request_id: str,
    input_payload: Mapping[str, object],
    dependencies: Mapping[str, object],
    tracer: Tracer | None,
    runner: Callable[[], ExecutionResult],
) -> ExecutionResult:
    """Execute one callable under standardized start/finish trace emission."""
    # Centralize trace lifecycle handling so individual pattern implementations stay focused on business logic.
    trace_scope = start_trace_run(
        agent_name=agent_name,
        request_id=request_id,
        input_payload=dict(input_payload),
        dependencies=dependencies,
        tracer=tracer,
    )
    try:
        result = runner()
    except Exception as exc:
        finish_trace_run(trace_scope, error=str(exc))
        raise
    result = enrich_execution_result_trace_metadata(result=result, tracer=tracer)
    finish_trace_run(trace_scope, result=result)
    return result


def attach_pattern_workflow(pattern: object, workflow: object) -> object:
    """Attach workflow runtime object to a pattern instance for diagnostics/tests."""
    cast(Any, pattern).workflow = workflow
    return workflow


def build_compiled_pattern_execution(
    *,
    workflow: Workflow,
    pattern_name: str,
    request_id: str,
    dependencies: Mapping[str, object],
    tracer: Tracer | None,
    input_payload: Mapping[str, object],
    workflow_request_id: str,
    finalize: Callable[[ExecutionResult], ExecutionResult],
    execution_mode: WorkflowExecutionMode = "sequential",
    failure_policy: WorkflowFailurePolicy = "skip_dependents",
) -> CompiledExecution:
    """Return a compiled execution wrapper for a workflow-backed pattern."""
    return CompiledExecution(
        workflow=workflow,
        input={},
        request_id=request_id,
        workflow_request_id=workflow_request_id,
        dependencies=dependencies,
        delegate_name=pattern_name,
        execution_mode=execution_mode,
        failure_policy=failure_policy,
        tracer=tracer,
        trace_input=dict(input_payload),
        finalize=finalize,
    )


def build_workflow_output_payload(workflow_result: ExecutionResult) -> dict[str, object]:
    """Return standardized workflow payload envelope for pattern outputs."""
    return {
        "workflow": workflow_result.to_dict(),
        "artifacts": workflow_result.output.get("artifacts", []),
    }


def resolve_prompt_override(
    *,
    override: str | None,
    default_value: str,
    field_name: str,
) -> str:
    """Resolve one prompt/template text using override-or-default semantics.

    Args:
        override: Optional user-provided prompt override.
        default_value: Default prompt text when no override is provided.
        field_name: Field label used for validation error messages.

    Returns:
        Validated prompt text.
    """
    if override is None:
        return validate_prompt_text(value=default_value, field_name=field_name)
    return validate_prompt_text(value=override, field_name=field_name)


def render_prompt_template(
    *,
    template_text: str,
    variables: Mapping[str, object],
    field_name: str,
) -> str:
    """Render template text with strict missing-variable validation.

    Args:
        template_text: Template text using ``string.Template`` placeholders.
        variables: Placeholder values used for template substitution.
        field_name: Field label used for validation error messages.

    Returns:
        Rendered prompt text.

    Raises:
        ValueError: If required template variables are missing.
    """
    normalized_template = validate_prompt_text(value=template_text, field_name=field_name)
    rendered_variables = {key: str(value) for key, value in variables.items()}
    template = Template(normalized_template)
    try:
        return template.substitute(rendered_variables)
    except KeyError as exc:
        missing_key = exc.args[0] if exc.args else "unknown"
        raise ValueError(f"{field_name} is missing required variable '{missing_key}'.") from exc


def attach_runtime_metadata(
    *,
    agent_result: ExecutionResult,
    requested_mode: str,
    resolved_mode: str,
    budget_metadata: Mapping[str, object],
    extra_metadata: Mapping[str, object] | None,
) -> ExecutionResult:
    """Attach standardized runtime metadata to an agent-style result payload.

    Args:
        agent_result: Base agent result to augment.
        requested_mode: Requested runtime mode name.
        resolved_mode: Effective runtime mode name.
        budget_metadata: Aggregated observed runtime metrics.
        extra_metadata: Optional additional runtime metadata sections.

    Returns:
        Agent result with normalized ``metadata["runtime"]`` payload.
    """
    metadata = dict(agent_result.metadata)
    # Keep runtime metadata under one namespaced key to avoid colliding with caller-defined metadata fields.
    runtime_metadata: dict[str, object] = {
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "observed_metrics": dict(budget_metadata),
    }
    if extra_metadata is not None:
        runtime_metadata.update(extra_metadata)
    metadata["runtime"] = runtime_metadata
    return ExecutionResult(
        output=dict(agent_result.output),
        success=agent_result.success,
        tool_results=list(agent_result.tool_results),
        model_response=agent_result.model_response,
        metadata=metadata,
    )


def build_pattern_failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
    tool_results: list[ToolResult] | None = None,
) -> ExecutionResult:
    """Build one normalized orchestration failure result.

    Args:
        error: Human-readable failure message.
        model_response: Optional model response associated with the failure.
        request_id: Request id for correlation/tracing.
        dependencies: Dependency mapping supplied to the orchestration run.
        metadata: Additional failure metadata fields.
        output: Structured failure output payload.
        tool_results: Optional tool results already produced before failure.

    Returns:
        Failure ``ExecutionResult`` with consistent metadata/output shape.
    """
    return build_failure_result(
        error=error,
        model_response=model_response,
        tool_results=tool_results or [],
        request_id=request_id,
        dependencies=dependencies,
        metadata=metadata,
        output=output,
    )
