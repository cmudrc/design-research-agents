"""Trace event emitters for model calls, tools, and decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._context import current_span_id, current_trace_session


def start_model_call(
    *,
    model: str,
    messages: Sequence[object],
    params: object,
    metadata: Mapping[str, object] | None = None,
) -> str | None:
    """Emit a model-call start event and return the span id.

    Args:
        model: Model identifier for the call.
        messages: Message payloads sent to the model.
        params: Provider-neutral generation params.
        metadata: Optional extra attributes to include on the span.

    Returns:
        Span id for the model call, or ``None`` when tracing is disabled.
    """
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
    """Emit a model-call completion event.

    Args:
        span_id: Span id for the model call.
        response: Optional response payload for success metadata.
        error: Optional error string for failure metadata.
        model: Optional model identifier when no response is available.
    """
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
    """Emit a model-call token event.

    Args:
        span_id: Span id for the model call.
        delta_text: Incremental text delta from the model.
    """
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
    """Emit a tool-call start event and return the span id.

    Args:
        tool_name: Tool identifier being invoked.
        tool_input: Tool input payload mapping.
        request_id: Request identifier for tracing.
        dependencies: Dependency payload mapping for the tool.

    Returns:
        Span id for the tool call, or ``None`` when tracing is disabled.
    """
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
    """Emit a tool-call completion event.

    Args:
        span_id: Span id for the tool call.
        tool_name: Tool identifier being invoked.
        result: Tool result payload or metadata.
    """
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
    """Emit a tool-call failure event.

    Args:
        span_id: Span id for the tool call.
        tool_name: Tool identifier being invoked.
        error: Error string describing the failure.
    """
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
    """Emit a router decision event.

    Args:
        source: Decision source label (e.g. "schema" or "fallback").
        alternatives: Available alternative tool names.
        selected_tool_name: Selected tool name, if any.
        selected_index: Selected alternative index, if any.
        reason: Optional reason text from the model or heuristic.
        parsed_route: Parsed route payload when available.
    """
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


def emit_model_selection_decision(
    *,
    model_id: str,
    provider: str,
    rationale: str,
    policy_id: str,
    policy_config: object,
    catalog_signature: str,
    intent: object,
    constraints: object,
    hardware_profile: object,
    candidate_count: int,
) -> None:
    """Emit a model selection decision event.

    Args:
        model_id: Selected model identifier.
        provider: Selected model provider.
        rationale: Selection rationale text.
        policy_id: Policy identifier used for selection.
        policy_config: Policy configuration used for selection.
        catalog_signature: Catalog signature used in selection.
        intent: Intent payload used for selection.
        constraints: Constraints payload used for selection.
        hardware_profile: Hardware profile snapshot used for selection.
        candidate_count: Number of candidates considered.
    """
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    session.emit_span_event(
        "ModelSelectionDecision",
        span_id=span_id,
        attributes={
            "model_id": model_id,
            "provider": provider,
            "rationale": rationale,
            "policy_id": policy_id,
            "policy_config": policy_config,
            "catalog_signature": catalog_signature,
            "intent": intent,
            "constraints": constraints,
            "hardware_profile": hardware_profile,
            "candidate_count": candidate_count,
        },
    )


def emit_tool_selection_decision(
    *,
    source: str,
    tool_name: str,
    reason: str,
    parsed_tool_call: Mapping[str, object] | None,
) -> None:
    """Emit a tool selection decision event.

    Args:
        source: Decision source label (e.g. "schema" or "fallback").
        tool_name: Selected tool name.
        reason: Optional reason text from the model or heuristic.
        parsed_tool_call: Parsed tool call payload when available.
    """
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
            "parsed_tool_call": (dict(parsed_tool_call) if parsed_tool_call is not None else None),
        },
    )


def emit_continuation_decision(
    *,
    step: int,
    should_continue: bool,
    reason: str,
    source: str,
) -> None:
    """Emit a continuation decision event.

    Args:
        step: Step index for the decision.
        should_continue: Decision outcome.
        reason: Optional reason text from the model or heuristic.
        source: Decision source label.
    """
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
    """Emit a guardrail decision event.

    Args:
        guardrail: Guardrail identifier.
        decision: Decision outcome label.
        reason: Optional reason text for the decision.
        details: Optional structured decision metadata.
    """
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
