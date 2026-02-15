"""Public tracing API for agent run instrumentation."""

from __future__ import annotations

from .config import TraceConfig, configure_tracing
from .context import TraceScope, current_trace_session, finish_trace_run, start_trace_run
from .emitters import (
    emit_continuation_decision,
    emit_guardrail_decision,
    emit_model_selection_decision,
    emit_model_token,
    emit_router_decision,
    emit_tool_selection_decision,
    fail_tool_call,
    finish_model_call,
    finish_tool_call,
    start_model_call,
    start_tool_call,
)
from .session import TraceEvent, TraceSession
from .sinks import ConsoleTraceSink, JSONLTraceSink, TraceSink

__all__ = [
    "ConsoleTraceSink",
    "JSONLTraceSink",
    "TraceConfig",
    "TraceEvent",
    "TraceScope",
    "TraceSession",
    "TraceSink",
    "configure_tracing",
    "current_trace_session",
    "emit_continuation_decision",
    "emit_guardrail_decision",
    "emit_model_selection_decision",
    "emit_model_token",
    "emit_router_decision",
    "emit_tool_selection_decision",
    "fail_tool_call",
    "finish_model_call",
    "finish_tool_call",
    "finish_trace_run",
    "start_model_call",
    "start_tool_call",
    "start_trace_run",
]
