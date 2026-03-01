"""Stable tracing exports for run instrumentation."""

from __future__ import annotations

from ._config import Tracer
from ._context import (
    TraceScope,
    current_trace_session,
    finish_trace_run,
    start_trace_run,
)
from ._emitters import (
    emit_continuation_decision,
    emit_guardrail_decision,
    emit_model_request_observed,
    emit_model_response_observed,
    emit_model_selection_decision,
    emit_model_token,
    emit_router_decision,
    emit_tool_invocation_observed,
    emit_tool_result_observed,
    emit_tool_selection_decision,
    emit_workflow_step_context,
    emit_workflow_step_result,
    fail_tool_call,
    finish_model_call,
    finish_tool_call,
    start_model_call,
    start_tool_call,
)
from ._session import TraceEvent, TraceSession
from ._sinks import ConsoleTraceSink, JSONLTraceSink, TraceSink

__all__ = [
    "ConsoleTraceSink",
    "JSONLTraceSink",
    "TraceEvent",
    "TraceScope",
    "TraceSession",
    "TraceSink",
    "Tracer",
    "current_trace_session",
    "emit_continuation_decision",
    "emit_guardrail_decision",
    "emit_model_request_observed",
    "emit_model_response_observed",
    "emit_model_selection_decision",
    "emit_model_token",
    "emit_router_decision",
    "emit_tool_invocation_observed",
    "emit_tool_result_observed",
    "emit_tool_selection_decision",
    "emit_workflow_step_context",
    "emit_workflow_step_result",
    "fail_tool_call",
    "finish_model_call",
    "finish_tool_call",
    "finish_trace_run",
    "start_model_call",
    "start_tool_call",
    "start_trace_run",
]
