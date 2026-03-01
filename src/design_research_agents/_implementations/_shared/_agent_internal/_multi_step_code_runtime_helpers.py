"""Runtime helpers for ``MultiStepCodeToolCallingAgent``."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._llm import LLMClient, LLMResponse
from design_research_agents._contracts._memory import MemoryStore
from design_research_agents._contracts._termination import (
    TERMINATED_COMPLETED,
    TERMINATED_MAX_STEPS_REACHED,
    TERMINATED_STEP_FAILURE,
)
from design_research_agents._contracts._tools import ToolRuntime, ToolSpec
from design_research_agents._implementations._shared._agent_internal._code_action_step_runner import (
    CodeActionStepRunner,
)
from design_research_agents._implementations._shared._agent_internal._input_parsing import (
    extract_boolean,
    extract_positive_int,
)
from design_research_agents._implementations._shared._agent_internal._multi_step_loop_state import (
    coerce_mapping,
    coerce_state_records,
    coerce_string_list,
    coerce_tool_results,
)
from design_research_agents._implementations._shared._agent_internal._multi_step_memory import (
    write_memory_observation,
)
from design_research_agents._implementations._shared._agent_internal._prompt_alternatives import (
    AlternativesPromptTarget,
)
from design_research_agents._implementations._shared._agent_internal._result_builders import (
    build_failure_result,
)
from design_research_agents._tracing import Tracer


@dataclass(slots=True, frozen=True, kw_only=True)
class MultiStepCodeRunConfig:
    """Resolved per-run configuration for multi-step code agent execution."""

    max_steps: int
    """Effective max-step limit for one run."""
    max_tool_calls_per_step: int
    """Effective per-step tool-call budget."""
    execution_timeout_seconds: int
    """Effective per-step sandbox timeout in seconds."""
    validate_tool_input_schema: bool
    """Whether tool input schema validation is enabled."""
    stop_on_step_failure: bool
    """Whether a failing step should terminate the loop."""


def resolve_run_config(
    *,
    input_payload: Mapping[str, object],
    defaults: MultiStepCodeRunConfig,
) -> MultiStepCodeRunConfig:
    """Resolve effective per-run settings from payload overrides.

    Args:
        input_payload: Normalized run input payload.
        defaults: Constructor-level default run configuration.

    Returns:
        Effective run configuration for this execution.
    """
    return MultiStepCodeRunConfig(
        max_steps=extract_positive_int(
            input_payload=input_payload,
            key="max_steps",
            default_value=defaults.max_steps,
        ),
        max_tool_calls_per_step=extract_positive_int(
            input_payload=input_payload,
            key="max_tool_calls_per_step",
            default_value=defaults.max_tool_calls_per_step,
        ),
        execution_timeout_seconds=extract_positive_int(
            input_payload=input_payload,
            key="execution_timeout_seconds",
            default_value=defaults.execution_timeout_seconds,
        ),
        validate_tool_input_schema=extract_boolean(
            input_payload=input_payload,
            key="validate_tool_input_schema",
            default_value=defaults.validate_tool_input_schema,
        ),
        stop_on_step_failure=extract_boolean(
            input_payload=input_payload,
            key="stop_on_step_failure",
            default_value=defaults.stop_on_step_failure,
        ),
    )


def build_code_step_agent(
    *,
    llm_client: LLMClient,
    tool_runtime: ToolRuntime,
    max_tool_calls_per_step: int,
    execution_timeout_seconds: int,
    validate_tool_input_schema: bool,
    normalize_generated_code_per_step: bool,
    default_tools_per_step: Sequence[Mapping[str, object]] | None,
    alternatives_prompt_target: AlternativesPromptTarget,
    tracer: Tracer | None,
) -> CodeActionStepRunner:
    """Build one configured step agent used by the multi-step code loop.

    Args:
        llm_client: LLM client dependency.
        tool_runtime: Tool runtime dependency.
        max_tool_calls_per_step: Per-step tool-call budget.
        execution_timeout_seconds: Per-step execution timeout.
        validate_tool_input_schema: Whether to enforce input schema checks.
        normalize_generated_code_per_step: Whether to normalize generated code each step.
        default_tools_per_step: Optional per-step allowed tools.
        alternatives_prompt_target: Alternatives prompt insertion target.
        tracer: Optional tracer dependency.

    Returns:
        Configured ``CodeActionStepRunner`` instance.
    """
    return CodeActionStepRunner(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_tool_calls=max_tool_calls_per_step,
        execution_timeout_seconds=execution_timeout_seconds,
        validate_tool_input_schema=validate_tool_input_schema,
        normalize_generated_code=normalize_generated_code_per_step,
        default_tools=default_tools_per_step,
        alternatives_prompt_target=alternatives_prompt_target,
        tracer=tracer,
    )


def build_step_tools_text(
    *,
    tool_specs: Mapping[str, ToolSpec],
    default_tools_per_step: Sequence[Mapping[str, object]] | None,
) -> str:
    """Build alternatives text describing allowed tools for each step.

    Args:
        tool_specs: Runtime tool specification mapping.
        default_tools_per_step: Optional per-step default tool allowlist.

    Returns:
        Formatted alternatives text block.
    """
    selected_specs: list[ToolSpec] = []
    if default_tools_per_step is None:
        selected_specs = list(tool_specs.values())
    else:
        for default_tool in default_tools_per_step:
            raw_name = default_tool.get("tool_name", default_tool.get("name"))
            if not isinstance(raw_name, str):
                continue
            normalized_name = raw_name.strip()
            if not normalized_name:
                continue
            runtime_spec = tool_specs.get(normalized_name)
            if runtime_spec is None:
                continue
            selected_specs.append(runtime_spec)

    tool_lines: list[str] = []
    for spec in selected_specs:
        tool_lines.append(
            "\n".join(
                [
                    f"- tool_name: {spec.name}",
                    f"  description: {spec.description or '(none)'}",
                    f"  input_schema: {json.dumps(spec.input_schema, sort_keys=True)}",
                ]
            )
        )
    return "\n".join(tool_lines)


def summarize_code_action(*, generated_code: object) -> str:
    """Return a compact generated-code summary for memory write-back.

    Args:
        generated_code: Generated code payload from one step.

    Returns:
        Compact generated-code summary string.
    """
    if not isinstance(generated_code, str):
        return ""
    normalized = generated_code.strip()
    if not normalized:
        return ""
    lines = normalized.splitlines()
    preview = " ".join(line.strip() for line in lines[:2] if line.strip())
    if len(preview) > 320:
        return preview[:317] + "..."
    return preview


def summarize_observation(*, final_output: object, error: object) -> str:
    """Return a compact observation summary for memory write-back.

    Args:
        final_output: Per-step final output payload.
        error: Per-step error payload.

    Returns:
        Compact observation summary string.
    """
    if isinstance(error, str) and error.strip():
        return f"error: {error.strip()}"
    if isinstance(final_output, Mapping):
        serialized = json.dumps(dict(final_output), ensure_ascii=True, sort_keys=True)
    else:
        serialized = str(final_output)
    normalized = serialized.strip()
    if len(normalized) > 320:
        return normalized[:317] + "..."
    return normalized


def resolve_step_completion(
    *,
    step_result: ExecutionResult,
    final_output: dict[str, object],
    stop_on_step_failure: bool,
) -> tuple[dict[str, object], str, bool, str | None, dict[str, object]]:
    """Resolve per-step termination state after one code step completes.

    Args:
        step_result: Step execution result.
        final_output: Current run-level final output mapping.
        stop_on_step_failure: Effective stop-on-failure setting.

    Returns:
        Tuple of final output, termination reason, continue flag, fatal error, and metadata.
    """
    terminated_reason = TERMINATED_MAX_STEPS_REACHED
    should_continue_next = True
    fatal_error: str | None = None
    fatal_metadata: dict[str, object] = {}

    if step_result.success:
        if bool(step_result.output.get("final_answer_called", False)):
            raw_final_output = step_result.output.get("final_output")
            if isinstance(raw_final_output, Mapping):
                final_output = dict(raw_final_output)
            terminated_reason = TERMINATED_COMPLETED
            should_continue_next = False
        return (
            final_output,
            terminated_reason,
            should_continue_next,
            fatal_error,
            fatal_metadata,
        )

    step_error = str(step_result.output.get("error", "Step execution failed."))
    terminated_reason = TERMINATED_STEP_FAILURE
    if stop_on_step_failure:
        should_continue_next = False
        fatal_error = step_error
        fatal_metadata = {
            "stage": "step_execution",
            "terminated_reason": terminated_reason,
        }
    return (
        final_output,
        terminated_reason,
        should_continue_next,
        fatal_error,
        fatal_metadata,
    )


def build_code_final_result(
    *,
    final_state: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
    max_steps: int,
    max_tool_calls_per_step: int,
    execution_timeout_seconds: int,
    validate_tool_input_schema: bool,
    normalize_generated_code_per_step: bool,
    stop_on_step_failure: bool,
    alternatives_prompt_target: str,
    step_memory_tail_items: int,
    memory_namespace: str,
    memory_read_top_k: int,
    memory_write_observations: bool,
    default_tools_per_step: Sequence[Mapping[str, object]] | None,
    memory_store_enabled: bool,
) -> ExecutionResult:
    """Build the final normalized result for the code multi-step agent.

    Args:
        final_state: Final loop-state mapping.
        request_id: Request identifier for the run.
        dependencies: Dependency payload used during the run.
        max_steps: Effective max-step setting.
        max_tool_calls_per_step: Effective per-step tool call cap.
        execution_timeout_seconds: Effective per-step timeout setting.
        validate_tool_input_schema: Effective validation flag.
        normalize_generated_code_per_step: Effective code normalization flag.
        stop_on_step_failure: Effective stop-on-failure setting.
        alternatives_prompt_target: Effective alternatives prompt target.
        step_memory_tail_items: Step memory tail item count.
        memory_namespace: Memory namespace used for read/write.
        memory_read_top_k: Memory retrieval top-k setting.
        memory_write_observations: Whether observation writes were enabled.
        default_tools_per_step: Optional default per-step tool allowlist.
        memory_store_enabled: Whether memory store dependency was configured.

    Returns:
        Final normalized execution result.
    """
    memory = coerce_state_records(final_state.get("memory"))
    decision_trace = coerce_state_records(final_state.get("decision_trace"))
    retrieval_trace = coerce_state_records(final_state.get("retrieval_trace"))
    memory_errors = coerce_string_list(final_state.get("memory_errors"))
    step_outputs = coerce_state_records(final_state.get("step_outputs"))
    tool_results = coerce_tool_results(final_state.get("tool_results"))
    final_output = coerce_mapping(final_state.get("final_output"))
    terminated_reason = str(final_state.get("terminated_reason", TERMINATED_MAX_STEPS_REACHED))
    maybe_model_response = final_state.get("last_model_response")
    last_model_response = maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None
    fatal_error = final_state.get("fatal_error")
    fatal_metadata = coerce_mapping(final_state.get("fatal_metadata"))
    if isinstance(fatal_error, str) and fatal_error:
        return build_failure_result(
            error=fatal_error,
            model_response=last_model_response,
            tool_results=tool_results,
            request_id=request_id,
            dependencies=dependencies,
            metadata={**fatal_metadata, "decision_trace": decision_trace},
            output={
                "final_output": final_output,
                "steps_executed": len(step_outputs),
                "step_outputs": step_outputs,
                "memory": memory,
                "terminated_reason": terminated_reason,
            },
        )

    success = terminated_reason == TERMINATED_COMPLETED
    return ExecutionResult(
        output={
            "final_output": final_output,
            "steps_executed": len(step_outputs),
            "step_outputs": step_outputs,
            "memory": memory,
            "terminated_reason": terminated_reason,
        },
        success=success,
        tool_results=tool_results,
        model_response=last_model_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            "decision_trace": decision_trace,
            "config": {
                "max_steps": max_steps,
                "max_tool_calls_per_step": max_tool_calls_per_step,
                "execution_timeout_seconds": execution_timeout_seconds,
                "validate_tool_input_schema": validate_tool_input_schema,
                "normalize_generated_code_per_step": normalize_generated_code_per_step,
                "stop_on_step_failure": stop_on_step_failure,
                "alternatives_prompt_target": alternatives_prompt_target,
                "step_memory_tail_items": step_memory_tail_items,
                "memory_namespace": memory_namespace,
                "memory_read_top_k": memory_read_top_k,
                "memory_write_observations": memory_write_observations,
                "default_tools_per_step": (
                    [dict(default_tool) for default_tool in default_tools_per_step]
                    if default_tools_per_step is not None
                    else None
                ),
            },
            "memory": {
                "enabled": memory_store_enabled,
                "retrieval_trace": retrieval_trace,
                "errors": memory_errors,
            },
        },
    )


def write_step_observation(
    *,
    memory_store: MemoryStore | None,
    namespace: str,
    task_prompt: str,
    step_number: int,
    step_reason: str,
    step_success: bool,
    generated_code: object,
    final_output: object,
    error: object,
) -> str | None:
    """Write one step observation summary to memory store when available.

    Args:
        memory_store: Optional memory store dependency.
        namespace: Memory namespace.
        task_prompt: Top-level task prompt text.
        step_number: One-based step number.
        step_reason: Optional thought/reason text for the step.
        step_success: Whether the step succeeded.
        generated_code: Generated code payload from the step.
        final_output: Final output payload from the step.
        error: Error payload from the step.

    Returns:
        Error text when memory write fails, otherwise ``None``.
    """
    return write_memory_observation(
        memory_store=memory_store,
        namespace=namespace,
        payload={
            "task": task_prompt,
            "step": step_number,
            "thought": step_reason,
            "selected_action": summarize_code_action(generated_code=generated_code),
            "observation_summary": summarize_observation(final_output=final_output, error=error),
            "success": step_success,
        },
        metadata={
            "kind": "multi_step_observation",
            "agent": "MultiStepCodeToolCallingAgent",
            "step": step_number,
            "success": step_success,
        },
    )


def append_retrieval_trace(
    *,
    retrieval_trace: list[dict[str, object]],
    memory_errors: list[str],
    retrieval_error: str | None,
    step_number: int,
    namespace: str,
    retrieved_match_count: int,
) -> None:
    """Append retrieval bookkeeping entries after one memory lookup.

    Args:
        retrieval_trace: Mutable retrieval trace list.
        memory_errors: Mutable memory error list.
        retrieval_error: Optional retrieval error text.
        step_number: One-based step number.
        namespace: Memory namespace used for retrieval.
        retrieved_match_count: Number of retrieved memory matches.
    """
    if retrieval_error is not None:
        memory_errors.append(f"read(step {step_number}): {retrieval_error}")
    retrieval_trace.append(
        {
            "step": step_number,
            "count": retrieved_match_count,
            "namespace": namespace,
        }
    )


def build_step_input(
    *,
    normalized_input: Mapping[str, object],
    step_prompt: str,
    max_tool_calls_per_step: int,
    execution_timeout_seconds: int,
    validate_tool_input_schema: bool,
    alternatives_prompt_target: str,
) -> dict[str, object]:
    """Build one normalized step-input payload for code step agents.

    Args:
        normalized_input: Base normalized run input payload.
        step_prompt: Rendered step prompt text.
        max_tool_calls_per_step: Per-step tool call cap.
        execution_timeout_seconds: Per-step execution timeout.
        validate_tool_input_schema: Whether to validate tool input schema.
        alternatives_prompt_target: Alternatives prompt insertion target.

    Returns:
        Step input payload mapping passed to ``CodeActionStepRunner``.
    """
    step_input = dict(normalized_input)
    step_input["prompt"] = step_prompt
    step_input["max_tool_calls"] = max_tool_calls_per_step
    step_input["execution_timeout_seconds"] = execution_timeout_seconds
    step_input["validate_tool_input_schema"] = validate_tool_input_schema
    step_input["alternatives_prompt_target"] = alternatives_prompt_target
    return step_input


__all__ = [
    "MultiStepCodeRunConfig",
    "append_retrieval_trace",
    "build_code_final_result",
    "build_code_step_agent",
    "build_step_input",
    "build_step_tools_text",
    "resolve_run_config",
    "resolve_step_completion",
    "summarize_code_action",
    "summarize_observation",
    "write_step_observation",
]
