"""Workflow step executors for tool, agent, logic, and memory step kinds."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import asdict
from typing import TypeGuard, cast

from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.memory import (
    MemorySearchQuery,
    MemoryStore,
    MemoryWriteRecord,
)
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import (
    AgentStep,
    LogicStep,
    MemoryReadStep,
    MemoryWriteStep,
    ToolStep,
    WorkflowDelegate,
    WorkflowDelegateRunner,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowStepResult,
)

from .step_context import build_invocation_dependencies, resolve_agent_prompt, resolve_tool_input


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
        if tool_result.error is not None:
            tool_error_message = tool_result.error.message
        else:
            tool_error_message = "Tool invocation failed."
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
    agents: Mapping[str, WorkflowDelegate],
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
        agents: Delegate registry for ``AgentStep`` resolution.
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
    selected_delegate = agents.get(step.agent_name)
    if selected_delegate is None:
        return _failed_step_result(
            step_id=step_id,
            error=f"Unknown agent '{step.agent_name}'.",
            metadata={"stage": "agent_binding", "agent_name": step.agent_name},
        )

    try:
        prompt = resolve_agent_prompt(step=step, step_context=step_context)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "agent_name": step.agent_name},
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

    if _is_workflow_delegate_runner(selected_delegate):
        nested_context = dict(step_context)
        nested_context["prompt"] = prompt
        try:
            workflow_result = selected_delegate.run(
                context=nested_context,
                execution_mode=execution_mode,
                failure_policy=failure_policy,
                request_id=request_scope,
                dependencies=invocation_dependencies,
            )
        except Exception as exc:
            return _failed_step_result(
                step_id=step_id,
                error=str(exc),
                metadata={
                    "stage": "execution",
                    "agent_name": step.agent_name,
                    "delegate_type": "workflow",
                },
            )

        serialized_output = workflow_result.asdict()
        if not workflow_result.success:
            return _failed_step_result(
                step_id=step_id,
                output=serialized_output,
                error="Nested workflow execution failed.",
                metadata={
                    "stage": "execution",
                    "agent_name": step.agent_name,
                    "delegate_type": "workflow",
                },
            )

        return WorkflowStepResult(
            step_id=step_id,
            status="completed",
            success=True,
            output=serialized_output,
            metadata={
                "stage": "execution",
                "agent_name": step.agent_name,
                "delegate_type": "workflow",
            },
        )

    selected_agent = cast(Agent, selected_delegate)
    try:
        agent_result = selected_agent.run(
            prompt,
            request_id=request_scope,
            dependencies=invocation_dependencies,
        )
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={
                "stage": "execution",
                "agent_name": step.agent_name,
                "delegate_type": "agent",
            },
        )

    serialized_output = agent_result.asdict()
    if not agent_result.success:
        return _failed_step_result(
            step_id=step_id,
            output=serialized_output,
            error=str(agent_result.output.get("error", "Agent execution failed.")),
            metadata={
                "stage": "execution",
                "agent_name": step.agent_name,
                "delegate_type": "agent",
            },
        )

    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=serialized_output,
        metadata={
            "stage": "execution",
            "agent_name": step.agent_name,
            "delegate_type": "agent",
        },
    )


def _is_workflow_delegate_runner(
    delegate: WorkflowDelegate,
) -> TypeGuard[WorkflowDelegateRunner]:
    """Return true when delegate ``run`` signature matches workflow style.

    Args:
        delegate: Delegate object registered for an ``AgentStep``.

    Returns:
        ``True`` when delegate ``run`` signature matches workflow-style execution.
    """
    run_callable = getattr(delegate, "run", None)
    if run_callable is None:
        return False
    try:
        signature = inspect.signature(run_callable)
    except (TypeError, ValueError):
        return False

    parameters = list(signature.parameters.values())
    if not parameters:
        return True

    first_parameter = parameters[0]
    return not (
        first_parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and first_parameter.name == "prompt"
    )


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
        if isinstance(raw_min_score, (int, float)):
            min_score = float(raw_min_score)
        else:
            min_score = step.min_score
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
