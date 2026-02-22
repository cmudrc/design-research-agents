"""Step-level tracing helpers for workflow execution spans."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from design_research_agents._contracts._workflow import (
    AgentStep,
    DelegateBatchStep,
    LoopStep,
    ModelStep,
    ToolStep,
    WorkflowStep,
)
from design_research_agents._tracing._context import (
    current_span_id,
    current_trace_session,
)


def step_kind(step: WorkflowStep) -> str:
    """Return a string label describing the workflow step kind.

    Args:
        step: Workflow step instance to classify.

    Returns:
        Stable step-kind label used in tracing attributes.
    """
    if isinstance(step, ToolStep):
        return "tool"
    if isinstance(step, AgentStep):
        return "agent"
    if isinstance(step, ModelStep):
        return "model"
    if isinstance(step, DelegateBatchStep):
        return "delegate_batch"
    if isinstance(step, LoopStep):
        return "loop"
    return "logic"


def start_step_span(*, step: WorkflowStep, step_id: str) -> str | None:
    """Start one step-level tracing span when a trace session is active.

    Args:
        step: Workflow step being executed.
        step_id: Runtime identifier for the step within the workflow run.

    Returns:
        Started span id, or ``None`` when tracing is inactive.
    """
    session = current_trace_session()
    if session is None:
        return None
    parent_span_id = current_span_id() or session.root_span_id
    return session.start_span(
        "WorkflowStepStarted",
        parent_span_id=parent_span_id,
        attributes={
            "step_id": step_id,
            "step_type": step_kind(step),
            "dependencies": list(step.dependencies),
        },
    )


def finish_step_span(*, span_id: str | None, step_id: str, status: str, error: str | None) -> None:
    """Finish one step-level tracing span when available.

    Args:
        span_id: Span id returned by :func:`start_step_span`.
        step_id: Runtime identifier for the finished step.
        status: Terminal step status (for example ``success`` or ``failed``).
        error: Optional error message recorded for failed steps.
    """
    session = current_trace_session()
    if session is None or span_id is None:
        return
    session.finish_span(
        "WorkflowStepFinished",
        span_id=span_id,
        attributes={
            "step_id": step_id,
            "status": status,
            "error": error,
        },
    )


@contextmanager
def activate_step_span(span_id: str | None) -> Iterator[None]:
    """Temporarily bind a step span as the active tracing span.

    Args:
        span_id: Span id to mark as current for nested tracing calls.

    Yields:
        Control to the wrapped execution block.
    """
    if span_id is None:
        yield
        return

    from design_research_agents._tracing import _context as tracing_context

    token = tracing_context._CURRENT_SPAN.set(span_id)
    try:
        yield
    finally:
        tracing_context._CURRENT_SPAN.reset(token)
