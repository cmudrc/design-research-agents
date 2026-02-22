"""Trace session context management."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass

from ._config import Tracer
from ._session import TraceSession, _SpanInfo
from ._utils import _normalize_value


@dataclass(slots=True)
class TraceScope:
    """Context manager payload for active trace scopes."""

    session: TraceSession
    """Field value for ``session``."""
    span_id: str
    """Field value for ``span_id``."""
    is_root: bool
    """Field value for ``is_root``."""
    trace_token: Token[TraceSession | None] | None = None
    """Field value for ``trace_token``."""
    span_token: Token[str | None] | None = None
    """Field value for ``span_token``."""
    agent_name: str | None = None
    """Field value for ``agent_name``."""


_CURRENT_TRACE: ContextVar[TraceSession | None] = ContextVar("dra_trace_session", default=None)
_CURRENT_SPAN: ContextVar[str | None] = ContextVar("dra_trace_span", default=None)


def current_trace_session() -> TraceSession | None:
    """Return the current trace session when active.

    Returns:
        Active trace session, or ``None`` when tracing is inactive.
    """
    return _CURRENT_TRACE.get()


def current_span_id() -> str | None:
    """Return the current active span id if one is set.

    Returns:
        Active span id, or ``None`` when no span is active.
    """
    return _CURRENT_SPAN.get()


def start_trace_run(
    *,
    agent_name: str,
    request_id: str,
    input_payload: Mapping[str, object],
    dependencies: Mapping[str, object],
    tracer: Tracer | None = None,
) -> TraceScope | None:
    """Start a trace run scope or nested agent span.

    Args:
        agent_name: Agent name for trace metadata.
        request_id: Request identifier for the run.
        input_payload: Normalized input payload mapping.
        dependencies: Dependency payload mapping.
        tracer: Explicit tracer dependency. When ``None`` tracing is disabled.

    Returns:
        Trace scope when tracing is enabled, otherwise ``None``.
    """
    if tracer is None or not tracer.enabled:
        return None

    active_session = _CURRENT_TRACE.get()
    if active_session is None:
        trace_path = tracer.build_trace_path(run_id=request_id)
        sinks = tracer.build_sinks(trace_path=trace_path)
        session = TraceSession(run_id=request_id, sinks=sinks)
        session._open_spans[session.root_span_id] = _SpanInfo(
            start_time=time.perf_counter(),
            parent_span_id=None,
        )
        trace_token = _CURRENT_TRACE.set(session)
        span_token = _CURRENT_SPAN.set(session.root_span_id)
        session.emit_event(
            "RunStarted",
            span_id=session.root_span_id,
            parent_span_id=None,
            attributes={
                "agent": agent_name,
                "request_id": request_id,
                "input": dict(input_payload),
                "dependency_keys": sorted(dependencies.keys()),
                "trace_path": str(trace_path) if trace_path is not None else None,
            },
        )
        return TraceScope(
            session=session,
            span_id=session.root_span_id,
            is_root=True,
            trace_token=trace_token,
            span_token=span_token,
            agent_name=agent_name,
        )

    parent_span_id = _CURRENT_SPAN.get() or active_session.root_span_id
    span_id = active_session.start_span(
        "AgentRunStarted",
        parent_span_id=parent_span_id,
        attributes={
            "agent": agent_name,
            "request_id": request_id,
            "input": dict(input_payload),
            "dependency_keys": sorted(dependencies.keys()),
        },
    )
    span_token = _CURRENT_SPAN.set(span_id)
    return TraceScope(
        session=active_session,
        span_id=span_id,
        is_root=False,
        span_token=span_token,
        agent_name=agent_name,
    )


def finish_trace_run(
    scope: TraceScope | None,
    *,
    result: object | None = None,
    error: str | None = None,
) -> None:
    """Finish a trace run scope with success or failure metadata.

    Args:
        scope: Trace scope returned by ``start_trace_run``.
        result: Optional result payload for success metadata.
        error: Optional error string for failure metadata.
    """
    if scope is None:
        return

    if scope.is_root:
        attributes = {
            "success": (bool(getattr(result, "success", False)) if result is not None else False),
            "error": error,
            "result": _normalize_value(result),
        }
        scope.session.finish_span(
            "RunFinished",
            span_id=scope.span_id,
            attributes=attributes,
        )
        scope.session.close()
        if scope.trace_token is not None:
            _CURRENT_TRACE.reset(scope.trace_token)
    else:
        attributes = {
            "success": (bool(getattr(result, "success", False)) if result is not None else False),
            "error": error,
            "agent": scope.agent_name,
        }
        scope.session.finish_span(
            "AgentRunFinished",
            span_id=scope.span_id,
            attributes=attributes,
        )

    if scope.span_token is not None:
        _CURRENT_SPAN.reset(scope.span_token)
