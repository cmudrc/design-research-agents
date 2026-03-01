"""Shared execution-context helpers for agent run methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from design_research_agents._implementations._shared._agent_internal._input_parsing import (
    extract_prompt,
)
from design_research_agents._implementations._shared._agent_internal._run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents._tracing import (
    Tracer,
    TraceScope,
    finish_trace_run,
    start_trace_run,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class AgentExecutionContext:
    """Normalized per-run context shared by agent implementations."""

    request_id: str
    """Resolved request identifier for one run."""
    dependencies: dict[str, object]
    """Normalized dependency payload mapping."""
    normalized_input: dict[str, object]
    """Normalized run input payload mapping."""
    prompt: str
    """Resolved prompt extracted from normalized input."""
    trace_scope: TraceScope | None
    """Optional active tracing scope for the run."""


def prepare_agent_execution(
    *,
    prompt: str,
    request_id: str | None,
    dependencies: Mapping[str, object] | None,
    agent_name: str,
    tracer: Tracer | None,
) -> AgentExecutionContext:
    """Resolve one run's request/dependencies/input payload and trace scope.

    Args:
        prompt: Raw prompt argument passed to ``Agent.run``.
        request_id: Optional caller-provided request identifier.
        dependencies: Optional dependency payload mapping.
        agent_name: Delegate name used for tracing events.
        tracer: Optional tracer dependency.

    Returns:
        Normalized execution context for the run.
    """
    execution_context = resolve_agent_execution_context(
        prompt=prompt,
        request_id=request_id,
        dependencies=dependencies,
    )
    trace_scope = start_trace_run(
        agent_name=agent_name,
        request_id=execution_context.request_id,
        input_payload=execution_context.normalized_input,
        dependencies=execution_context.dependencies,
        tracer=tracer,
    )
    return AgentExecutionContext(
        request_id=execution_context.request_id,
        dependencies=execution_context.dependencies,
        normalized_input=execution_context.normalized_input,
        prompt=execution_context.prompt,
        trace_scope=trace_scope,
    )


def resolve_agent_execution_context(
    *,
    prompt: str,
    request_id: str | None,
    dependencies: Mapping[str, object] | None,
) -> AgentExecutionContext:
    """Resolve one run's request/dependencies/input payload without tracing."""
    resolved_request_id = resolve_request_id(request_id)
    resolved_dependencies = normalize_dependencies(dependencies)
    normalized_input = normalize_input_payload(prompt)
    return AgentExecutionContext(
        request_id=resolved_request_id,
        dependencies=resolved_dependencies,
        normalized_input=normalized_input,
        prompt=extract_prompt(normalized_input),
        trace_scope=None,
    )


def finish_agent_execution(
    *,
    trace_scope: TraceScope | None,
    result: object | None = None,
    error: str | None = None,
) -> None:
    """Finish a run trace scope with either result or error metadata.

    Args:
        trace_scope: Scope returned by ``prepare_agent_execution``.
        result: Optional execution result payload.
        error: Optional error message.
    """
    finish_trace_run(trace_scope, result=result, error=error)


__all__ = [
    "AgentExecutionContext",
    "finish_agent_execution",
    "prepare_agent_execution",
    "resolve_agent_execution_context",
]
