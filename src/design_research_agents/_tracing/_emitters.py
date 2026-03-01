"""Trace event emitters for model calls, tools, and decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._context import current_span_id, current_trace_session
from ._utils import _sanitize_trace_payload


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
    # Attach model spans under the current span when nested, otherwise under run root.
    parent_span_id = current_span_id() or session.root_span_id
    attributes = {
        "model": model,
        "message_count": len(messages),
        "messages": _sanitize_trace_payload(list(messages)),
        "params": _sanitize_trace_payload(params),
    }
    if metadata:
        sanitized_metadata = _sanitize_trace_payload(dict(metadata))
        if isinstance(sanitized_metadata, Mapping):
            attributes.update(dict(sanitized_metadata))
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
        # Preserve partial response metadata when available to aid postmortem debugging.
        model_name = getattr(response, "model", None) if response is not None else model
        session.finish_span(
            "ModelCallFailed",
            span_id=span_id,
            attributes={
                "error": error,
                "model": model_name,
                "usage": _sanitize_trace_payload(getattr(response, "usage", None) if response is not None else None),
                "latency_ms": _sanitize_trace_payload(
                    getattr(response, "latency_ms", None) if response is not None else None
                ),
            },
        )
        return
    session.finish_span(
        "ModelCallFinished",
        span_id=span_id,
        attributes={
            "response": _sanitize_trace_payload(response),
            "model": getattr(response, "model", None),
            "usage": _sanitize_trace_payload(getattr(response, "usage", None)),
            "latency_ms": _sanitize_trace_payload(getattr(response, "latency_ms", None)),
            "finish_reason": _sanitize_trace_payload(getattr(response, "finish_reason", None)),
            "provider": _sanitize_trace_payload(getattr(response, "provider", None)),
        },
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


def emit_model_request_observed(
    *,
    source: str,
    model: str | None,
    request_payload: object,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Emit one model request-observation event on the current span.

    Args:
        source: Source label for where the request was observed.
        model: Model identifier when available.
        request_payload: Raw request payload observed at this boundary.
        metadata: Optional extra attributes to include.
    """
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    attributes: dict[str, object] = {
        "source": source,
        "model": model,
        # Sanitize request payload before persistence to avoid leaking secrets/large blobs.
        "request": _sanitize_trace_payload(request_payload),
    }
    if metadata:
        attributes["metadata"] = _sanitize_trace_payload(dict(metadata))
    session.emit_span_event(
        "ModelRequestObserved",
        span_id=span_id,
        attributes=attributes,
    )


def emit_model_response_observed(
    *,
    source: str,
    response_payload: object | None = None,
    error: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Emit one model response-observation event on the current span.

    Args:
        source: Source label for where the response was observed.
        response_payload: Raw response payload observed at this boundary.
        error: Optional error text when the call failed.
        metadata: Optional extra attributes to include.
    """
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    attributes: dict[str, object] = {
        "source": source,
        "response": _sanitize_trace_payload(response_payload),
        "error": error,
    }
    if response_payload is not None:
        attributes["usage"] = _sanitize_trace_payload(getattr(response_payload, "usage", None))
        attributes["latency_ms"] = _sanitize_trace_payload(getattr(response_payload, "latency_ms", None))
        attributes["finish_reason"] = _sanitize_trace_payload(getattr(response_payload, "finish_reason", None))
        attributes["provider"] = _sanitize_trace_payload(getattr(response_payload, "provider", None))
        attributes["model"] = _sanitize_trace_payload(getattr(response_payload, "model", None))
    if metadata:
        attributes["metadata"] = _sanitize_trace_payload(dict(metadata))
    session.emit_span_event(
        "ModelResponseObserved",
        span_id=span_id,
        attributes=attributes,
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
            "tool_input": _sanitize_trace_payload(dict(tool_input)),
            "request_id": request_id,
            # Only persist dependency keys here to keep span payloads compact.
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
            "ok": True,
            "result": _sanitize_trace_payload(result),
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
        attributes={"tool_name": tool_name, "ok": False, "error": error},
    )


def emit_tool_invocation_observed(
    *,
    tool_name: str,
    source_id: str | None,
    request_id: str,
    tool_input: Mapping[str, object],
    dependency_keys: Sequence[str],
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Emit one tool invocation-observation event on the current span.

    Args:
        tool_name: Routed tool name.
        source_id: Source identifier where invocation is routed.
        request_id: Invocation request id.
        tool_input: Tool input payload observed before invocation.
        dependency_keys: Sorted dependency keys visible to the tool.
        metadata: Optional extra attributes to include.
    """
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    attributes: dict[str, object] = {
        "tool_name": tool_name,
        "source_id": source_id,
        "request_id": request_id,
        "tool_input": _sanitize_trace_payload(dict(tool_input)),
        "dependency_keys": list(dependency_keys),
    }
    if metadata:
        attributes["metadata"] = _sanitize_trace_payload(dict(metadata))
    session.emit_span_event(
        "ToolInvocationObserved",
        span_id=span_id,
        attributes=attributes,
    )


def emit_tool_result_observed(
    *,
    tool_name: str,
    source_id: str | None,
    ok: bool | None,
    result_payload: object | None = None,
    error: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Emit one tool result-observation event on the current span.

    Args:
        tool_name: Routed tool name.
        source_id: Source identifier where invocation was routed.
        ok: Tool success flag when available.
        result_payload: Tool result payload or partial metadata.
        error: Optional failure message.
        metadata: Optional extra attributes to include.
    """
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    attributes: dict[str, object] = {
        "tool_name": tool_name,
        "source_id": source_id,
        "ok": ok,
        # Result payloads may contain arbitrary objects; sanitize before emission.
        "result": _sanitize_trace_payload(result_payload),
        "error": error,
    }
    if metadata:
        attributes["metadata"] = _sanitize_trace_payload(dict(metadata))
    session.emit_span_event(
        "ToolResultObserved",
        span_id=span_id,
        attributes=attributes,
    )


def emit_workflow_step_context(
    *,
    step_id: str,
    step_type: str,
    context: Mapping[str, object],
) -> None:
    """Emit workflow step context observed before execution.

    Args:
        step_id: Step identifier.
        step_type: Step-kind label.
        context: Full context mapping visible to the step.
    """
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    # Emit pre-execution context snapshots to make step-input debugging reproducible.
    session.emit_span_event(
        "WorkflowStepContextObserved",
        span_id=span_id,
        attributes={
            "step_id": step_id,
            "step_type": step_type,
            "context": _sanitize_trace_payload(dict(context)),
        },
    )


def emit_workflow_step_result(
    *,
    step_id: str,
    step_type: str,
    status: str,
    success: bool,
    output: Mapping[str, object],
    error: str | None,
    metadata: Mapping[str, object],
) -> None:
    """Emit workflow step result observed after execution.

    Args:
        step_id: Step identifier.
        step_type: Step-kind label.
        status: Result status.
        success: Result success flag.
        output: Step output payload.
        error: Optional error text.
        metadata: Result metadata payload.
    """
    session = current_trace_session()
    if session is None:
        return
    span_id = current_span_id() or session.root_span_id
    # Emit normalized post-execution payload for status/error analysis tooling.
    session.emit_span_event(
        "WorkflowStepResultObserved",
        span_id=span_id,
        attributes={
            "step_id": step_id,
            "step_type": step_type,
            "status": status,
            "success": success,
            "output": _sanitize_trace_payload(dict(output)),
            "error": error,
            "metadata": _sanitize_trace_payload(dict(metadata)),
        },
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
    # Keep full decision context in trace to make selection rationale auditable after the fact.
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
        # Optional details payload enables policy-specific diagnostics without schema churn.
        attributes["details"] = dict(details)
    session.emit_span_event(
        "GuardrailDecision",
        span_id=span_id,
        attributes=attributes,
    )
