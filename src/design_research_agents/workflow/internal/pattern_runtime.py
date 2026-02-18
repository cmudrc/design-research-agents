"""Shared helpers for workflow-native orchestration patterns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from string import Template

from design_research_agents.agent.internal.prompt_overrides import validate_prompt_text
from design_research_agents.agent.internal.result_builders import build_failure_result
from design_research_agents.contracts.agent import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.tools import ToolResult, ToolSpec


@dataclass(slots=True)
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
