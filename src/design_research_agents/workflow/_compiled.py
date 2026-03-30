"""Compiled workflow execution wrapper for delegate compile APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._workflow import (
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
)
from design_research_agents._tracing import Tracer, finish_trace_run, start_trace_run
from design_research_agents._tracing._result_metadata import (
    enrich_execution_result_trace_metadata,
)

from .workflow import Workflow


def _identity_result(result: ExecutionResult) -> ExecutionResult:
    """Return the provided workflow result unchanged."""
    return result


@dataclass(slots=True, frozen=True, kw_only=True)
class CompiledExecution:
    """Bound compiled delegate execution that can be run repeatedly."""

    workflow: Workflow
    """Workflow graph compiled for this execution."""
    input: str | Mapping[str, object] | None
    """Bound workflow input payload."""
    request_id: str
    """Top-level request identifier for delegate tracing."""
    dependencies: Mapping[str, object]
    """Bound dependency payload mapping."""
    delegate_name: str
    """Delegate name used for top-level trace metadata."""
    finalize: Callable[[ExecutionResult], ExecutionResult] = _identity_result
    """Finalizer that maps the raw workflow result into the delegate result."""
    execution_mode: WorkflowExecutionMode = "sequential"
    """Workflow execution mode used by ``run()``."""
    failure_policy: WorkflowFailurePolicy = "skip_dependents"
    """Workflow failure policy used by ``run()``."""
    tracer: Tracer | None = None
    """Optional tracer used for top-level compile-run traces."""
    trace_input: Mapping[str, object] = field(default_factory=dict)
    """Input payload attached to the top-level trace scope."""
    workflow_request_id: str | None = None
    """Optional nested workflow request id override."""

    def to_mermaid(self, *, direction: str = "TD") -> str:
        """Return a Mermaid diagram for the bound compiled workflow."""
        return self.workflow.to_mermaid(direction=direction)

    def to_svg(self, *, direction: str = "TD") -> str:
        """Return an SVG diagram for the bound compiled workflow."""
        return self.workflow.to_svg(direction=direction)

    def run(self) -> ExecutionResult:
        """Execute the compiled workflow and finalize the result."""
        trace_scope = start_trace_run(
            agent_name=self.delegate_name,
            request_id=self.request_id,
            input_payload=dict(self.trace_input),
            dependencies=dict(self.dependencies),
            tracer=self.tracer,
        )
        try:
            workflow_result = self.workflow.run(
                input=self.input,
                execution_mode=self.execution_mode,
                failure_policy=self.failure_policy,
                request_id=self.workflow_request_id or self.request_id,
                dependencies=self.dependencies,
            )
            result = self.finalize(workflow_result)
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise
        result = enrich_execution_result_trace_metadata(result=result, tracer=self.tracer)
        finish_trace_run(trace_scope, result=result)
        return result


__all__ = ["CompiledExecution"]
