"""Multi-step ReAct-style agent built as a loop over internal JSON action steps.

The agent alternates continuation checks with step execution, recording a
structured thought-action-observation memory trace and aggregating tool
results across steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import LLMClient, LLMResponse
from design_research_agents.contracts.memory import MemoryStore
from design_research_agents.contracts.termination import (
    SOURCE_INVALID_PAYLOAD,
    TERMINATED_CONTINUATION_INVALID_PAYLOAD,
    TERMINATED_MAX_STEPS_REACHED,
    TERMINATED_STEP_FAILURE,
    continuation_stopped_reason,
)
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.tracing import Tracer

from ..execution_context import (
    finish_agent_execution,
    prepare_agent_execution,
)
from ..input_parsing import (
    extract_boolean as _extract_boolean,
)
from ..input_parsing import (
    extract_positive_int as _extract_positive_int,
)
from ..json_action_step_runner import (
    JsonActionStepRunner,
)
from ..model_resolution import (
    resolve_agent_model,
)
from ..multi_step_common import (
    build_step_prompt,
)
from ..multi_step_continuation import (
    llm_should_continue as _llm_should_continue,
)
from ..multi_step_json_helpers import (
    build_step_tools_text,
    normalize_step_final_output,
    resolve_step_error,
)
from ..multi_step_json_runtime_helpers import (
    build_json_final_result,
)
from ..multi_step_json_runtime_helpers import (
    summarize_observation as _summarize_observation,
)
from ..multi_step_json_runtime_helpers import (
    summarize_tool_action as _summarize_tool_action,
)
from ..multi_step_loop_state import (
    build_loop_initial_state,
    continue_loop,
)
from ..multi_step_loop_state import (
    coerce_mapping as _coerce_mapping,
)
from ..multi_step_loop_state import (
    coerce_state_records as _coerce_state_records,
)
from ..multi_step_loop_state import (
    coerce_string_list as _coerce_string_list,
)
from ..multi_step_loop_state import (
    coerce_tool_results as _coerce_tool_results,
)
from ..multi_step_memory import (
    retrieve_memory_context,
    write_memory_observation,
)
from ..prompt_alternatives import (
    AlternativesPromptTarget,
    normalize_alternatives_prompt_target,
)
from ..prompt_overrides import (
    resolve_prompt_text,
)
from ..response_schemas import (
    build_continuation_response_schema,
)
from ..workflow_loop_orchestration import (
    run_workflow_loop,
)


class MultiStepJsonToolCallingAgent(Agent):
    """Agent that iterates action-observation steps until continuation stops.

    Each iteration asks the model whether to continue, then delegates one action
    step to ``JsonActionStepRunner``. The loop keeps explicit
    ReAct-style thought-action-observation entries in memory.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        max_steps: int = 5,
        stop_on_step_failure: bool = True,
        continuation_system_prompt: str | None = None,
        continuation_user_prompt_template: str | None = None,
        step_user_prompt_template: str | None = None,
        tool_calling_system_prompt: str | None = None,
        tool_calling_user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        continuation_memory_tail_items: int = 6,
        step_memory_tail_items: int = 8,
        memory_store: MemoryStore | None = None,
        memory_namespace: str = "default",
        memory_read_top_k: int = 4,
        memory_write_observations: bool = True,
        allowed_tools: Sequence[str] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a multi-step JSON tool-calling agent.

        Args:
            llm_client: LLM client used for continuation and step generation.
            tool_runtime: Tool runtime shared across all steps.
            max_steps: Maximum number of action-observation iterations.
            stop_on_step_failure: Whether to stop immediately when one step fails.
            continuation_system_prompt: Optional continuation system prompt override.
            continuation_user_prompt_template: Optional continuation user prompt template.
            step_user_prompt_template: Optional step user prompt template.
            tool_calling_system_prompt: Optional system prompt for tool selection step.
            tool_calling_user_prompt_template: Optional user template for tool selection step.
            alternatives_prompt_target: Prompt target for alternatives blocks.
            continuation_memory_tail_items: Memory tail size for continuation prompts.
            step_memory_tail_items: Memory tail size for step prompts.
            memory_store: Optional persistent memory store for retrieval/write-back.
            memory_namespace: Namespace partition used for memory reads/writes.
            memory_read_top_k: Number of memory matches retrieved per step.
            memory_write_observations: Whether to persist per-step observations.
            allowed_tools: Optional tool allowlist used by action steps.
            tracer: Optional explicit tracer dependency.

        Raises:
            ValueError: If ``max_steps``, memory tail items, or ``memory_read_top_k`` are
                less than ``1``.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if continuation_memory_tail_items < 1:
            raise ValueError("continuation_memory_tail_items must be >= 1.")
        if step_memory_tail_items < 1:
            raise ValueError("step_memory_tail_items must be >= 1.")
        if memory_read_top_k < 1:
            raise ValueError("memory_read_top_k must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self.workflow: object | None = None
        self._max_steps = max_steps
        self._stop_on_step_failure = stop_on_step_failure
        self._continuation_system_prompt = resolve_prompt_text(
            override=continuation_system_prompt,
            default_prompt_name="multi_step_continue_system",
            field_name="continuation_system_prompt",
        )
        self._continuation_user_prompt_template = resolve_prompt_text(
            override=continuation_user_prompt_template,
            default_prompt_name="multi_step_continue_user",
            field_name="continuation_user_prompt_template",
        )
        self._step_user_prompt_template = resolve_prompt_text(
            override=step_user_prompt_template,
            default_prompt_name="multi_step_json_step_user",
            field_name="step_user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(
            alternatives_prompt_target
        )
        self._tool_calling_system_prompt = tool_calling_system_prompt
        self._tool_calling_user_prompt_template = tool_calling_user_prompt_template
        self._continuation_memory_tail_items = continuation_memory_tail_items
        self._step_memory_tail_items = step_memory_tail_items
        self._memory_store = memory_store
        self._memory_namespace = memory_namespace.strip() or "default"
        self._memory_read_top_k = memory_read_top_k
        self._memory_write_observations = memory_write_observations
        self._allowed_tools = tuple(allowed_tools) if allowed_tools is not None else None
        self._continuation_response_schema = build_continuation_response_schema()

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run the multi-step action-observation loop and return aggregated results.

        The run collects continuation decisions, per-step outputs, and all tool
        results while preserving memory entries that can be inspected by callers.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.

        Raises:
            RuntimeError: Propagates failures from continuation/model/tool execution
                helpers while running the loop.
        """
        execution_context = prepare_agent_execution(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            agent_name="MultiStepJsonToolCallingAgent",
            tracer=self._tracer,
        )
        resolved_request_id = execution_context.request_id
        resolved_dependencies = execution_context.dependencies
        prompt = execution_context.prompt
        max_steps = _extract_positive_int(
            input_payload=execution_context.normalized_input,
            key="max_steps",
            default_value=self._max_steps,
        )
        stop_on_step_failure = _extract_boolean(
            input_payload=execution_context.normalized_input,
            key="stop_on_step_failure",
            default_value=self._stop_on_step_failure,
        )
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        alternatives_prompt_target = self._alternatives_prompt_target
        step_tools_text = build_step_tools_text(
            tool_specs={spec.name: spec for spec in self._tool_runtime.list_tools()},
        )

        step_agent = JsonActionStepRunner(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            system_prompt=self._tool_calling_system_prompt,
            user_prompt_template=self._tool_calling_user_prompt_template,
            alternatives_prompt_target=alternatives_prompt_target,
            allowed_tools=self._allowed_tools,
            tracer=self._tracer,
        )

        try:
            loop_result = run_workflow_loop(
                max_iterations=max_steps,
                initial_state=build_loop_initial_state(
                    prompt=prompt,
                    include_continuation=True,
                ),
                continue_predicate=continue_loop,
                iteration_handler=lambda iteration, state: self._run_loop_iteration(
                    iteration=iteration,
                    state=state,
                    prompt=prompt,
                    max_steps=max_steps,
                    resolved_model=resolved_model,
                    alternatives_prompt_target=alternatives_prompt_target,
                    step_tools_text=step_tools_text,
                    step_agent=step_agent,
                    request_id=resolved_request_id,
                    dependencies=resolved_dependencies,
                    stop_on_step_failure=stop_on_step_failure,
                ),
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                tracer=self._tracer,
            )
        except Exception as exc:
            finish_agent_execution(trace_scope=execution_context.trace_scope, error=str(exc))
            raise
        self.workflow = loop_result.workflow
        result = build_json_final_result(
            final_state=loop_result.final_state,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
            max_steps=max_steps,
            stop_on_step_failure=stop_on_step_failure,
            alternatives_prompt_target=alternatives_prompt_target,
            continuation_memory_tail_items=self._continuation_memory_tail_items,
            step_memory_tail_items=self._step_memory_tail_items,
            memory_namespace=self._memory_namespace,
            memory_read_top_k=self._memory_read_top_k,
            memory_write_observations=self._memory_write_observations,
            memory_store_enabled=self._memory_store is not None,
        )
        merged_output = dict(result.output)
        merged_output["workflow"] = loop_result.workflow_result.asdict()
        merged_output["artifacts"] = loop_result.workflow_result.output.get("artifacts", [])
        result = ExecutionResult(
            output=merged_output,
            success=result.success,
            tool_results=list(result.tool_results),
            model_response=result.model_response,
            metadata=dict(result.metadata),
            step_results=dict(result.step_results),
            execution_order=list(result.execution_order),
        )
        finish_agent_execution(trace_scope=execution_context.trace_scope, result=result)
        return result

    def _run_loop_iteration(
        self,
        *,
        iteration: int,
        state: Mapping[str, object],
        prompt: str,
        max_steps: int,
        resolved_model: str,
        alternatives_prompt_target: AlternativesPromptTarget,
        step_tools_text: str,
        step_agent: JsonActionStepRunner,
        request_id: str,
        dependencies: Mapping[str, object],
        stop_on_step_failure: bool,
    ) -> Mapping[str, object]:
        """Execute one JSON tool loop iteration and produce next loop state.

        Args:
            iteration: One-based loop iteration number.
            state: Current loop-state mapping.
            prompt: User prompt text.
            max_steps: Effective max-step limit.
            resolved_model: Resolved model identifier.
            alternatives_prompt_target: Prompt target for alternatives injection.
            step_tools_text: Alternatives/tool block text.
            step_agent: Step-level JSON tool agent instance.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency payload mapping.
            stop_on_step_failure: Effective stop-on-failure setting.

        Returns:
            Next loop-state mapping.
        """
        step_number = iteration
        step_index = iteration - 1
        memory = _coerce_state_records(state.get("memory"))
        continuation_trace = _coerce_state_records(state.get("continuation_trace"))
        retrieval_trace = _coerce_state_records(state.get("retrieval_trace"))
        memory_errors = _coerce_string_list(state.get("memory_errors"))
        step_outputs = _coerce_state_records(state.get("step_outputs"))
        tool_results = _coerce_tool_results(state.get("tool_results"))
        final_output = _coerce_mapping(state.get("final_output"))
        maybe_model_response = state.get("last_model_response")
        last_model_response = (
            maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None
        )

        retrieved_context, retrieved_matches, retrieval_error = retrieve_memory_context(
            memory_store=self._memory_store,
            namespace=self._memory_namespace,
            top_k=self._memory_read_top_k,
            task_prompt=prompt,
            memory=memory,
            memory_tail_items=self._continuation_memory_tail_items,
        )
        if retrieval_error is not None:
            memory_errors.append(f"read(step {step_number}): {retrieval_error}")
        retrieval_trace.append(
            {
                "step": step_number,
                "count": len(retrieved_matches),
                "namespace": self._memory_namespace,
            }
        )

        skip_continuation_probe = max_steps == 1 and step_number == 1

        if skip_continuation_probe:
            should_continue = True
            continue_reason = "single_step_mode"
            continue_source = "single_step_mode"
            continue_response = None
        else:
            (
                should_continue,
                continue_reason,
                continue_source,
                continue_response,
            ) = _llm_should_continue(
                llm_client=self._llm_client,
                prompt=prompt,
                memory=memory,
                step_index=step_index,
                max_steps=max_steps,
                model=resolved_model,
                alternatives_prompt_target=alternatives_prompt_target,
                alternatives_text=step_tools_text,
                retrieved_context=retrieved_context,
                continuation_system_prompt=self._continuation_system_prompt,
                continuation_user_prompt_template=self._continuation_user_prompt_template,
                continuation_response_schema=self._continuation_response_schema,
                continuation_memory_tail_items=self._continuation_memory_tail_items,
                alternatives_section_label="Available tools for action steps",
                agent_name="MultiStepJsonToolCallingAgent",
            )
            if continue_response is not None:
                last_model_response = continue_response
        continuation_trace.append(
            {
                "step": step_number,
                "continue": should_continue,
                "thought": continue_reason,
                "reason": continue_reason,
                "source": continue_source,
            }
        )
        memory.append(
            {
                "kind": "thought",
                "step": step_number,
                "continue": should_continue,
                "text": continue_reason,
                "source": continue_source,
            }
        )
        if not should_continue:
            terminated_reason = (
                TERMINATED_CONTINUATION_INVALID_PAYLOAD
                if continue_source == SOURCE_INVALID_PAYLOAD
                else continuation_stopped_reason(continue_source)
            )
            continuation_fatal_error: str | None = None
            continuation_fatal_metadata: dict[str, object] = {}
            if continue_source == SOURCE_INVALID_PAYLOAD:
                continuation_fatal_error = (
                    "Continuation output was invalid. Expected JSON payload with "
                    "boolean `continue`."
                )
                continuation_fatal_metadata = {
                    "stage": "continuation",
                    "terminated_reason": terminated_reason,
                }
            return {
                "memory": memory,
                "continuation_trace": continuation_trace,
                "retrieval_trace": retrieval_trace,
                "memory_errors": memory_errors,
                "step_outputs": step_outputs,
                "tool_results": tool_results,
                "final_output": final_output,
                "last_model_response": last_model_response,
                "terminated_reason": terminated_reason,
                "should_continue": False,
                "fatal_error": continuation_fatal_error,
                "fatal_metadata": continuation_fatal_metadata,
            }

        step_prompt = build_step_prompt(
            prompt=prompt,
            memory=memory,
            step_number=step_number,
            prompt_template=self._step_user_prompt_template,
            memory_tail_items=self._step_memory_tail_items,
            retrieved_context=retrieved_context,
        )
        step_result = step_agent.run(
            step_prompt,
            request_id=f"{request_id}:step-{step_number}",
            dependencies=dependencies,
        )
        if step_result.model_response is not None:
            last_model_response = step_result.model_response

        tool_results.extend(step_result.tool_results)
        raw_tool_output = step_result.output.get("tool_output")
        step_final_output = normalize_step_final_output(raw_tool_output)
        step_error = resolve_step_error(step_result)
        step_outputs.append(
            {
                "step": step_number,
                "success": step_result.success,
                "final_output": step_final_output,
                "tool_name": step_result.output.get("tool_name"),
                "tool_input": step_result.output.get("tool_input", {}),
                "error": step_error,
                "tool_results_count": len(step_result.tool_results),
            }
        )
        memory.extend(
            [
                {
                    "kind": "action",
                    "step": step_number,
                    "tool_name": step_result.output.get("tool_name"),
                    "tool_input": step_result.output.get("tool_input", {}),
                },
                {
                    "kind": "observation",
                    "step": step_number,
                    "success": step_result.success,
                    "final_output": step_final_output,
                    "error": step_error,
                },
            ]
        )

        if self._memory_write_observations:
            memory_write_error = write_memory_observation(
                memory_store=self._memory_store,
                namespace=self._memory_namespace,
                payload={
                    "task": prompt,
                    "step": step_number,
                    "thought": continue_reason,
                    "selected_action": _summarize_tool_action(
                        tool_name=step_result.output.get("tool_name"),
                        tool_input=step_result.output.get("tool_input"),
                    ),
                    "observation_summary": _summarize_observation(
                        final_output=step_final_output,
                        error=step_error,
                    ),
                    "success": step_result.success,
                },
                metadata={
                    "kind": "multi_step_observation",
                    "agent": "MultiStepJsonToolCallingAgent",
                    "step": step_number,
                    "success": step_result.success,
                },
            )
            if memory_write_error is not None:
                memory_errors.append(f"write(step {step_number}): {memory_write_error}")

        terminated_reason = TERMINATED_MAX_STEPS_REACHED
        should_continue_next = True
        fatal_error: str | None = None
        fatal_metadata: dict[str, object] = {}
        if step_result.success:
            final_output = step_final_output
        else:
            terminated_reason = TERMINATED_STEP_FAILURE
            if stop_on_step_failure:
                should_continue_next = False
                fatal_error = step_error
                fatal_metadata = {
                    "stage": "step_execution",
                    "terminated_reason": terminated_reason,
                }

        return {
            "memory": memory,
            "continuation_trace": continuation_trace,
            "retrieval_trace": retrieval_trace,
            "memory_errors": memory_errors,
            "step_outputs": step_outputs,
            "tool_results": tool_results,
            "final_output": final_output,
            "last_model_response": last_model_response,
            "terminated_reason": terminated_reason,
            "should_continue": should_continue_next,
            "fatal_error": fatal_error,
            "fatal_metadata": fatal_metadata,
        }


__all__ = [
    "MultiStepJsonToolCallingAgent",
]
