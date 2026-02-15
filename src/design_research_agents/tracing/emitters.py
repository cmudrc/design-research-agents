"""Trace event emitters for model calls, tools, and decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .context import current_span_id, current_trace_session


def start_model_call(
    *,
    model: str,
    messages: Sequence[object],
    params: object,
    metadata: Mapping[str, object] | None = None,
) -> str | None:
    """Emit a model-call start event and return the span id."""
    session = current_trace_session()
    if session is None:
        return None
    parent_span_id = current_span_id() or session.root_span_id
    attributes = {
        "model": model,
        "message_count": len(messages),
        "messages": messages,
        "params": params,
    }
    if metadata:
        attributes.update(dict(metadata))
    return session.start_span(
        "ModelCallStarted",
        parent_span_id=parent_span_id,
        attributes=attributes,
    )


def finish_model_call(
    span_id: str | None,
    *,
    response: object | None = None,
    error: str | None = None,
    model: str | None = None,
) -> None:
    """Emit a model-call completion event."""
    session = current_trace_session()
    if session is None or span_id is None:
        return
    if error is not None:
        model_name = getattr(response, "model", None) if response is not None else model
        session.finish_span(
            "ModelCallFailed",
            span_id=span_id,
            attributes={"error": error, "model": model_name},
        )
        return
    session.finish_span(
        "ModelCallFinished",
        span_id=span_id,
        attributes={"response": response, "model": getattr(response, "model", None)},
    )


def emit_model_token(span_id: str | None, *, delta_text: str) -> None:
    """Emit a model-call token event."""
    session = current_trace_session()
    if session is None or span_id is None:
        return
    if not delta_text:
        return
    session.emit_span_event(
        "ModelCallToken",
        span_id=span_id,
        attributes={"delta_text": delta_text},
    )


def start_tool_call(
    *,
    tool_name: str,
    tool_input: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> str | None:
    """Emit a tool-call start event and return the span id."""
    session = current_trace_session()
    if session is None:
        return None
    parent_span_id = current_span_id() or session.root_span_id
    return session.start_span(
        "ToolCallStarted",
        parent_span_id=parent_span_id,
        attributes={
            "tool_name": tool_name,
            "tool_input": dict(tool_input),
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
        },
    )


def finish_tool_call(
    span_id: str | None,
    *,
    tool_name: str,
    result: object | None,
) -> None:
    """Emit a tool-call completion event."""
    session = current_trace_session()
    if session is None or span_id is None:
        return
    session.finish_span(
        "ToolCallFinished",
        span_id=span_id,
        attributes={
            "tool_name": tool_name,
            "result": result,
        },
    )


def fail_tool_call(
    span_id: str | None,
    *,
    tool_name: str,
    error: str,
) -> None:
    """Emit a tool-call failure event."""
    session = current_trace_session()
    if session is None or span_id is None:
        return
    session.finish_span(
        "ToolCallFailed",
        span_id=span_id,
        attributes={"tool_name": tool_name, "error": error},
    )


def emit_router_decision(
    *,
    source: str,
    alternatives: list[str],
    selected_tool_name: str | None,
    selected_index: int | None,
    reason: str | None,
    parsed_route: Mapping[str, object] | None,
) -> None:
    """Emit a router decision event."""
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    session.emit_span_event(
        "RouterDecision",
        span_id=span_id,
        attributes={
            "source": source,
            "alternatives": list(alternatives),
            "selected_tool_name": selected_tool_name,
            "selected_index": selected_index,
            "reason": reason,
            "parsed_route": dict(parsed_route) if parsed_route is not None else None,
        },
    )


def emit_tool_selection_decision(
    *,
    source: str,
    tool_name: str,
    reason: str,
    parsed_tool_call: Mapping[str, object] | None,
) -> None:
    """Emit a tool selection decision event."""
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    session.emit_span_event(
        "ToolSelectionDecision",
        span_id=span_id,
        attributes={
            "source": source,
            "tool_name": tool_name,
            "reason": reason,
            "parsed_tool_call": dict(parsed_tool_call) if parsed_tool_call is not None else None,
        },
    )


def emit_continuation_decision(
    *,
    step: int,
    should_continue: bool,
    reason: str,
    source: str,
) -> None:
    """Emit a continuation decision event."""
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    session.emit_span_event(
        "ContinuationDecision",
        span_id=span_id,
        attributes={
            "step": step,
            "continue": should_continue,
            "reason": reason,
            "source": source,
        },
    )


def emit_guardrail_decision(
    *,
    guardrail: str,
    decision: str,
    reason: str,
    details: Mapping[str, object] | None = None,
) -> None:
    """Emit a guardrail decision event."""
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    attributes: dict[str, object] = {
        "guardrail": guardrail,
        "decision": decision,
        "reason": reason,
    }
    if details:
        attributes["details"] = dict(details)
    session.emit_span_event(
        "GuardrailDecision",
        span_id=span_id,
        attributes=attributes,
    )
