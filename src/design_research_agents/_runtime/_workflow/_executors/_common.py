"""Workflow step executors for tool, agent, logic, and memory step kinds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace

from design_research_agents._contracts._llm import LLMChatParams, LLMRequest, LLMResponse
from design_research_agents._contracts._memory import (
    MemorySearchQuery,
    MemoryStore,
    MemoryWriteRecord,
)
from design_research_agents._contracts._tools import ToolRuntime
from design_research_agents._contracts._workflow import (
    AgentStep,
    DelegateBatchCall,
    DelegateBatchStep,
    LogicStep,
    MemoryReadStep,
    MemoryWriteStep,
    ModelStep,
    ToolStep,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowStepResult,
)
from design_research_agents._runtime._common._delegate_invocation import invoke_delegate
from design_research_agents._tracing import (
    emit_model_request_observed,
    emit_model_response_observed,
)

from .._step_context import (
    build_invocation_dependencies,
    resolve_agent_prompt,
    resolve_tool_input,
)


def run_tool_step(
    *,
    tool_runtime: ToolRuntime | None,
    step: ToolStep,
    step_id: str,
    step_context: Mapping[str, object],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one tool step and return normalized workflow step result.

    Args:
        tool_runtime: Tool runtime used to resolve and invoke tools.
        step: Tool step definition to execute.
        step_id: Step identifier for result metadata.
        step_context: Step execution context with dependency outputs.
        request_id: Workflow request id for scoped tool invocation ids.
        execution_mode: Effective workflow execution mode.
        failure_policy: Effective workflow failure policy.
        dependencies: Run dependency mapping passed into tool invocation.

    Returns:
        Normalized workflow step result for this tool step.
    """
    if tool_runtime is None:
        return _failed_step_result(
            step_id=step_id,
            error="Tool step requires a configured tool_runtime.",
            metadata={"stage": "tool_binding", "tool_name": step.tool_name},
        )

    available_tools = {tool_spec.name for tool_spec in tool_runtime.list_tools()}
    if step.tool_name not in available_tools:
        return _failed_step_result(
            step_id=step_id,
            error=f"Unknown tool '{step.tool_name}'.",
            metadata={"stage": "tool_binding", "tool_name": step.tool_name},
        )

    try:
        tool_input = resolve_tool_input(step=step, step_context=step_context)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "tool_name": step.tool_name},
        )

    invocation_dependencies = build_invocation_dependencies(
        base_dependencies=dependencies,
        step_id=step_id,
        request_id=request_id,
        execution_mode=execution_mode,
        failure_policy=failure_policy,
        step_context=step_context,
    )

    try:
        tool_result = tool_runtime.invoke(
            step.tool_name,
            tool_input,
            request_id=f"{request_id}:workflow:{step_id}",
            dependencies=invocation_dependencies,
        )
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "execution", "tool_name": step.tool_name},
        )

    serialized_output = asdict(tool_result)
    if not tool_result.ok:
        tool_error_message = tool_result.error.message if tool_result.error is not None else "Tool invocation failed."
        return _failed_step_result(
            step_id=step_id,
            output=serialized_output,
            error=tool_error_message,
            metadata={"stage": "execution", "tool_name": step.tool_name},
        )

    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=serialized_output,
        metadata={"stage": "execution", "tool_name": step.tool_name},
    )


def run_agent_step(
    *,
    step: AgentStep,
    step_id: str,
    step_context: Mapping[str, object],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one agent-like step and return normalized workflow step result.

    Args:
        step: Agent step definition to execute.
        step_id: Step identifier for result metadata.
        step_context: Step execution context with dependency outputs.
        request_id: Workflow request id for scoped delegate invocation ids.
        execution_mode: Effective workflow execution mode.
        failure_policy: Effective workflow failure policy.
        dependencies: Run dependency mapping passed into delegate invocation.

    Returns:
        Normalized workflow step result for this agent step.
    """
    selected_delegate = step.delegate

    try:
        prompt = resolve_agent_prompt(step=step, step_context=step_context)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "delegate_class": type(selected_delegate).__name__},
        )

    invocation_dependencies = build_invocation_dependencies(
        base_dependencies=dependencies,
        step_id=step_id,
        request_id=request_id,
        execution_mode=execution_mode,
        failure_policy=failure_policy,
        step_context=step_context,
    )

    request_scope = f"{request_id}:workflow:{step_id}"
    try:
        delegate_invocation = invoke_delegate(
            delegate=selected_delegate,
            prompt=prompt,
            step_context=step_context,
            request_id=request_scope,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            dependencies=invocation_dependencies,
        )
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={
                "stage": "execution",
                "delegate_class": type(selected_delegate).__name__,
                "delegate_type": "unknown",
            },
        )

    agent_result = delegate_invocation.result
    serialized_output = agent_result.asdict()
    if not agent_result.success:
        if delegate_invocation.delegate_type == "workflow":
            error_message = "Nested workflow execution failed."
        else:
            error_message = str(agent_result.output.get("error", "Agent execution failed."))
        return _failed_step_result(
            step_id=step_id,
            output=serialized_output,
            error=error_message,
            metadata={
                "stage": "execution",
                "delegate_class": type(selected_delegate).__name__,
                "delegate_type": delegate_invocation.delegate_type,
            },
        )

    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=serialized_output,
        metadata={
            "stage": "execution",
            "delegate_class": type(selected_delegate).__name__,
            "delegate_type": delegate_invocation.delegate_type,
        },
    )


def run_model_step(
    *,
    step: ModelStep,
    step_id: str,
    step_context: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one model step and return normalized workflow step result."""
    try:
        llm_request = _build_model_request(step=step, step_context=step_context)
        llm_request = _apply_terminal_output_schema(
            llm_request=llm_request,
            step_context=step_context,
        )
        emit_model_request_observed(
            source="WorkflowRuntime.run_model_step",
            model=llm_request.model,
            request_payload=llm_request,
            metadata={"step_id": step_id},
        )
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "step_kind": "model"},
        )

    try:
        model_response = _execute_model_request(step=step, llm_request=llm_request)
        emit_model_response_observed(
            source="WorkflowRuntime.run_model_step",
            response_payload=model_response,
            metadata={"step_id": step_id},
        )
    except Exception as exc:
        emit_model_response_observed(
            source="WorkflowRuntime.run_model_step",
            error=str(exc),
            metadata={"step_id": step_id},
        )
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "execution", "step_kind": "model"},
        )

    try:
        parsed_payload = _parse_model_response_payload(
            step=step,
            model_response=model_response,
            step_context=step_context,
        )
    except Exception as exc:
        emit_model_response_observed(
            source="WorkflowRuntime.run_model_step.parse",
            response_payload=model_response,
            error=str(exc),
            metadata={"step_id": step_id},
        )
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "response_parse", "step_kind": "model"},
        )

    output: dict[str, object] = {
        "model_response": asdict(model_response),
        "parsed": parsed_payload,
        "final_output": _resolve_model_final_output(
            parsed_payload=parsed_payload,
            model_response=model_response,
        ),
    }
    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=output,
        metadata={"stage": "execution", "step_kind": "model"},
    )


def _apply_terminal_output_schema(
    *,
    llm_request: LLMRequest,
    step_context: Mapping[str, object],
) -> LLMRequest:
    """Apply workflow output schema to terminal model requests when unset."""
    workflow_context = step_context.get("_workflow")
    if not isinstance(workflow_context, Mapping):
        return llm_request
    if not bool(workflow_context.get("is_terminal_step", False)):
        return llm_request
    if llm_request.response_schema is not None:
        return llm_request
    raw_output_schema = workflow_context.get("output_schema")
    if not isinstance(raw_output_schema, Mapping):
        return llm_request
    return replace(llm_request, response_schema=dict(raw_output_schema))


def run_delegate_batch_step(
    *,
    step: DelegateBatchStep,
    step_id: str,
    step_context: Mapping[str, object],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one delegate-batch step and return normalized workflow step result."""
    try:
        raw_calls = step.calls_builder(step_context)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "step_kind": "delegate_batch"},
        )
    try:
        normalized_calls = _normalize_delegate_batch_calls(raw_calls=raw_calls)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "step_kind": "delegate_batch"},
        )

    invocation_dependencies = build_invocation_dependencies(
        base_dependencies=dependencies,
        step_id=step_id,
        request_id=request_id,
        execution_mode=execution_mode,
        failure_policy=failure_policy,
        step_context=step_context,
    )
    results, failed_call_id = _execute_delegate_batch_calls(
        calls=normalized_calls,
        step_context=step_context,
        request_id=request_id,
        step_id=step_id,
        invocation_dependencies=invocation_dependencies,
        fail_fast=step.fail_fast,
    )

    all_success = failed_call_id is None
    output = {
        "results": results,
        "all_success": all_success,
        "failed_call_id": failed_call_id,
        "final_output": _resolve_delegate_batch_final_output(results),
    }
    if not all_success:
        assert failed_call_id is not None
        return _failed_step_result(
            step_id=step_id,
            output=output,
            error=f"Delegate batch call '{failed_call_id}' failed.",
            metadata={
                "stage": "execution",
                "step_kind": "delegate_batch",
                "fail_fast": step.fail_fast,
            },
        )
    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=output,
        metadata={"stage": "execution", "step_kind": "delegate_batch", "fail_fast": step.fail_fast},
    )


def _build_model_request(
    *,
    step: ModelStep,
    step_context: Mapping[str, object],
) -> LLMRequest:
    """Build and validate ``ModelStep`` request payload."""
    llm_request = step.request_builder(step_context)
    if not isinstance(llm_request, LLMRequest):
        raise TypeError("ModelStep request_builder must return an LLMRequest.")
    return llm_request


def _resolve_model_request_model_id(*, llm_client: object, llm_request: LLMRequest) -> str:
    """Resolve model id for ``chat``-based model-step execution."""
    if llm_request.model is not None:
        return llm_request.model

    default_model = getattr(llm_client, "default_model", None)
    if not callable(default_model):
        raise TypeError("ModelStep llm_client requires request.model or callable default_model().")
    return str(default_model())


def _execute_model_request(*, step: ModelStep, llm_request: LLMRequest) -> LLMResponse:
    """Execute one model request using ``generate`` or ``chat`` fallback."""
    generate_callable = getattr(step.llm_client, "generate", None)
    if callable(generate_callable):
        generated_response = generate_callable(llm_request)
        if not isinstance(generated_response, LLMResponse):
            raise TypeError("ModelStep llm_client.generate() must return an LLMResponse.")
        return generated_response

    chat_callable = getattr(step.llm_client, "chat", None)
    if not callable(chat_callable):
        raise TypeError("ModelStep llm_client must expose generate() or chat().")

    chat_response = chat_callable(
        llm_request.messages,
        model=_resolve_model_request_model_id(llm_client=step.llm_client, llm_request=llm_request),
        params=LLMChatParams(
            temperature=llm_request.temperature,
            max_tokens=llm_request.max_tokens,
            response_schema=llm_request.response_schema,
            provider_options=dict(llm_request.provider_options),
        ),
    )
    if not isinstance(chat_response, LLMResponse):
        raise TypeError("ModelStep llm_client.chat() must return an LLMResponse.")
    return chat_response


def _parse_model_response_payload(
    *,
    step: ModelStep,
    model_response: LLMResponse,
    step_context: Mapping[str, object],
) -> dict[str, object]:
    """Parse model-step response payload into normalized mapping output."""
    if step.response_parser is None:
        return {"model_text": model_response.text}

    parsed_output = step.response_parser(model_response, step_context)
    if not isinstance(parsed_output, Mapping):
        raise TypeError("ModelStep response_parser must return a mapping.")
    return dict(parsed_output)


def _resolve_model_final_output(
    *,
    parsed_payload: Mapping[str, object],
    model_response: LLMResponse,
) -> dict[str, object]:
    """Resolve canonical ``final_output`` payload for one model step."""
    final_output_raw = parsed_payload.get("final_output")
    if isinstance(final_output_raw, Mapping):
        return dict(final_output_raw)
    if parsed_payload:
        return dict(parsed_payload)
    return {"model_text": model_response.text}


def _normalize_delegate_batch_calls(*, raw_calls: object) -> list[DelegateBatchCall]:
    """Normalize and validate delegate-batch calls-builder output."""
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise TypeError("DelegateBatchStep calls_builder must return a sequence.")

    normalized_calls: list[DelegateBatchCall] = []
    seen_call_ids: set[str] = set()
    for index, raw_call in enumerate(raw_calls, start=1):
        normalized_call = _normalize_delegate_batch_call(raw_call, index=index)
        if normalized_call.call_id in seen_call_ids:
            raise ValueError(f"Duplicate delegate batch call_id '{normalized_call.call_id}'.")
        seen_call_ids.add(normalized_call.call_id)
        normalized_calls.append(normalized_call)
    return normalized_calls


def _run_delegate_batch_call(
    *,
    call: DelegateBatchCall,
    step_context: Mapping[str, object],
    request_id: str,
    step_id: str,
    invocation_dependencies: Mapping[str, object],
) -> dict[str, object]:
    """Invoke one delegate-batch call and normalize result payload."""
    call_request_id = f"{request_id}:workflow:{step_id}:delegate_batch:{call.call_id}"
    try:
        delegate_invocation = invoke_delegate(
            delegate=call.delegate,
            prompt=call.prompt,
            step_context=step_context,
            request_id=call_request_id,
            execution_mode=call.execution_mode,
            failure_policy=call.failure_policy,
            dependencies=invocation_dependencies,
        )
        call_result = delegate_invocation.result
    except Exception as exc:
        return {
            "call_id": call.call_id,
            "success": False,
            "output": {},
            "error": str(exc),
            "metadata": {},
            "delegate_type": "unknown",
            "model_response": None,
            "tool_results": [],
        }

    call_error = str(call_result.output.get("error", "")) if not call_result.success else None
    return {
        "call_id": call.call_id,
        "success": call_result.success,
        "output": dict(call_result.output),
        "error": call_error,
        "metadata": dict(call_result.metadata),
        "delegate_type": delegate_invocation.delegate_type,
        "model_response": call_result.model_response,
        "tool_results": list(call_result.tool_results),
    }


def _execute_delegate_batch_calls(
    *,
    calls: Sequence[DelegateBatchCall],
    step_context: Mapping[str, object],
    request_id: str,
    step_id: str,
    invocation_dependencies: Mapping[str, object],
    fail_fast: bool,
) -> tuple[list[dict[str, object]], str | None]:
    """Execute normalized delegate-batch calls and collect call result entries."""
    results: list[dict[str, object]] = []
    failed_call_id: str | None = None
    for call in calls:
        call_result_entry = _run_delegate_batch_call(
            call=call,
            step_context=step_context,
            request_id=request_id,
            step_id=step_id,
            invocation_dependencies=invocation_dependencies,
        )
        results.append(call_result_entry)
        if not bool(call_result_entry.get("success")) and failed_call_id is None:
            failed_call_id = call.call_id
        if failed_call_id is not None and fail_fast:
            break
    return results, failed_call_id


def _resolve_delegate_batch_final_output(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Extract canonical delegate-batch ``final_output`` from call results."""
    if not results:
        return {}
    last_output = results[-1].get("output")
    if not isinstance(last_output, Mapping):
        return {}
    maybe_final_output = last_output.get("final_output")
    if isinstance(maybe_final_output, Mapping):
        return dict(maybe_final_output)
    return dict(last_output)


def run_logic_step(
    *,
    step: LogicStep,
    step_id: str,
    step_context: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one logic step handler and normalize result payload.

    Args:
        step: Logic step definition to execute.
        step_id: Step identifier for result metadata.
        step_context: Step execution context with dependency outputs.

    Returns:
        Normalized workflow step result for this logic step.
    """
    try:
        step_output = dict(step.handler(step_context))
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "execution"},
        )

    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=step_output,
        metadata={"stage": "execution"},
    )


def run_memory_read_step(
    *,
    memory_store: MemoryStore | None,
    step: MemoryReadStep,
    step_id: str,
    step_context: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one memory read step and return normalized workflow step result.

    Args:
        memory_store: Memory store used for retrieval.
        step: Memory-read step definition to execute.
        step_id: Step identifier for result metadata.
        step_context: Step execution context with dependency outputs.

    Returns:
        Normalized workflow step result for this memory-read step.
    """
    if memory_store is None:
        return _failed_step_result(
            step_id=step_id,
            error="Memory step requires a configured memory_store.",
            metadata={"stage": "memory_binding", "step_kind": "memory_read"},
        )

    try:
        built_query = step.query_builder(step_context)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "step_kind": "memory_read"},
        )

    try:
        memory_query = _normalize_memory_search_query(step=step, built_query=built_query)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "step_kind": "memory_read"},
        )

    try:
        matches = memory_store.search(memory_query)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "execution", "step_kind": "memory_read"},
        )

    output = {
        "query": memory_query.asdict(),
        "matches": [record.asdict() for record in matches],
        "count": len(matches),
        "namespace": memory_query.namespace,
    }
    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=output,
        metadata={"stage": "execution", "step_kind": "memory_read"},
    )


def run_memory_write_step(
    *,
    memory_store: MemoryStore | None,
    step: MemoryWriteStep,
    step_id: str,
    step_context: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one memory write step and return normalized workflow step result.

    Args:
        memory_store: Memory store used for persistence.
        step: Memory-write step definition to execute.
        step_id: Step identifier for result metadata.
        step_context: Step execution context with dependency outputs.

    Returns:
        Normalized workflow step result for this memory-write step.
    """
    if memory_store is None:
        return _failed_step_result(
            step_id=step_id,
            error="Memory step requires a configured memory_store.",
            metadata={"stage": "memory_binding", "step_kind": "memory_write"},
        )

    try:
        built_records = step.records_builder(step_context)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "step_kind": "memory_write"},
        )

    try:
        normalized_records = _normalize_memory_write_records(built_records)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "step_kind": "memory_write"},
        )

    try:
        written_records = memory_store.write(normalized_records, namespace=step.namespace)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "execution", "step_kind": "memory_write"},
        )

    output = {
        "written": len(written_records),
        "namespace": step.namespace,
        "ids": [record.item_id for record in written_records],
    }
    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=output,
        metadata={"stage": "execution", "step_kind": "memory_write"},
    )


def _normalize_memory_search_query(
    *,
    step: MemoryReadStep,
    built_query: str | Mapping[str, object],
) -> MemorySearchQuery:
    """Normalize ``MemoryReadStep`` query-builder output into query contract.

    Args:
        step: Memory read step configuration.
        built_query: Query-builder output payload.

    Returns:
        Normalized memory search query.

    Raises:
        TypeError: Raised when query-builder output type is unsupported.
    """
    if isinstance(built_query, str):
        query_text = built_query
        metadata_filters: dict[str, object] = {}
        top_k = step.top_k
        min_score = step.min_score
    elif isinstance(built_query, Mapping):
        text_value = built_query.get("text", built_query.get("query", ""))
        query_text = str(text_value)

        raw_filters = built_query.get("metadata_filters")
        metadata_filters = dict(raw_filters) if isinstance(raw_filters, Mapping) else {}

        raw_top_k = built_query.get("top_k")
        top_k = raw_top_k if isinstance(raw_top_k, int) else step.top_k

        raw_min_score = built_query.get("min_score")
        min_score = float(raw_min_score) if isinstance(raw_min_score, (int, float)) else step.min_score
    else:
        raise TypeError("MemoryReadStep query_builder must return a string or mapping.")

    normalized_top_k = max(1, int(top_k))
    return MemorySearchQuery(
        text=query_text,
        namespace=step.namespace,
        top_k=normalized_top_k,
        min_score=min_score,
        metadata_filters=metadata_filters,
    )


def _normalize_memory_write_records(
    built_records: object,
) -> list[MemoryWriteRecord]:
    """Normalize write-builder output into ``MemoryWriteRecord`` list.

    Args:
        built_records: Write-builder output payload.

    Returns:
        Normalized write records.

    Raises:
        TypeError: Raised for unsupported write record payload types.
        ValueError: Raised when mapping payloads omit ``content``.
    """
    if not isinstance(built_records, (list, tuple)):
        raise TypeError("MemoryWriteStep records_builder must return a sequence.")

    normalized_records: list[MemoryWriteRecord] = []
    for record in built_records:
        if isinstance(record, MemoryWriteRecord):
            normalized_records.append(record)
            continue

        if isinstance(record, str):
            normalized_records.append(MemoryWriteRecord(content=record))
            continue

        if isinstance(record, Mapping):
            raw_content = record.get("content")
            if raw_content is None:
                raise ValueError("Memory write records must include 'content'.")
            raw_metadata = record.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            raw_item_id = record.get("item_id")
            item_id = str(raw_item_id).strip() if isinstance(raw_item_id, str) else None
            normalized_records.append(
                MemoryWriteRecord(
                    content=str(raw_content),
                    metadata=metadata,
                    item_id=item_id or None,
                )
            )
            continue

        raise TypeError("Unsupported memory write record type.")

    return normalized_records


def _normalize_delegate_batch_call(raw_call: object, *, index: int) -> DelegateBatchCall:
    """Normalize one delegate-batch call specification."""
    if isinstance(raw_call, DelegateBatchCall):
        return raw_call
    if not isinstance(raw_call, Mapping):
        raise TypeError("DelegateBatchStep calls must be DelegateBatchCall or mapping payloads.")

    raw_call_id = raw_call.get("call_id", f"call_{index}")
    call_id = str(raw_call_id).strip()
    if not call_id:
        raise ValueError("DelegateBatchStep call_id must be non-empty.")
    if "delegate" not in raw_call:
        raise ValueError(f"DelegateBatchStep call '{call_id}' is missing delegate.")
    if "prompt" not in raw_call:
        raise ValueError(f"DelegateBatchStep call '{call_id}' is missing prompt.")
    raw_prompt = raw_call.get("prompt")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError(f"DelegateBatchStep call '{call_id}' prompt must be a non-empty string.")

    raw_execution_mode = raw_call.get("execution_mode", "sequential")
    execution_mode: WorkflowExecutionMode = (
        raw_execution_mode if raw_execution_mode in {"sequential", "dag"} else "sequential"
    )
    raw_failure_policy = raw_call.get("failure_policy", "skip_dependents")
    normalized_failure_policy: WorkflowFailurePolicy = (
        raw_failure_policy if raw_failure_policy in {"skip_dependents", "propagate_failed_state"} else "skip_dependents"
    )
    return DelegateBatchCall(
        call_id=call_id,
        delegate=raw_call["delegate"],
        prompt=raw_prompt.strip(),
        execution_mode=execution_mode,
        failure_policy=normalized_failure_policy,
    )


def _failed_step_result(
    *,
    step_id: str,
    error: str,
    metadata: Mapping[str, object],
    output: Mapping[str, object] | None = None,
) -> WorkflowStepResult:
    """Build a standardized failed ``WorkflowStepResult`` payload.

    Args:
        step_id: Step identifier for the failed step.
        error: Human-readable failure message.
        metadata: Additional metadata to attach to the result.
        output: Optional partial output captured before failure.

    Returns:
        Failed workflow step result.
    """
    return WorkflowStepResult(
        step_id=step_id,
        status="failed",
        success=False,
        output=dict(output or {}),
        error=error,
        metadata=dict(metadata),
    )
