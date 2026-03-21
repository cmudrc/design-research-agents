"""Multi-step ReAct-style agent built as a loop over internal JSON action steps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._llm import LLMClient, LLMResponse
from design_research_agents._contracts._memory import MemoryStore
from design_research_agents._contracts._termination import (
    TERMINATED_COMPLETED,
    TERMINATED_MAX_STEPS_REACHED,
    TERMINATED_STEP_FAILURE,
)
from design_research_agents._contracts._tools import ToolRuntime
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution

from .._execution_context import (
    resolve_agent_execution_context,
)
from .._input_parsing import (
    extract_boolean as _extract_boolean,
)
from .._input_parsing import (
    extract_positive_int as _extract_positive_int,
)
from .._json_action_step_runner import (
    JsonActionStepRunner,
)
from .._multi_step_common import (
    build_step_prompt,
)
from .._multi_step_json_helpers import (
    normalize_step_final_output,
    resolve_step_error,
)
from .._multi_step_json_runtime_helpers import (
    build_json_final_result,
)
from .._multi_step_json_runtime_helpers import (
    summarize_observation as _summarize_observation,
)
from .._multi_step_json_runtime_helpers import (
    summarize_tool_action as _summarize_tool_action,
)
from .._multi_step_loop_state import (
    build_loop_initial_state,
    continue_loop,
)
from .._multi_step_loop_state import (
    coerce_mapping as _coerce_mapping,
)
from .._multi_step_loop_state import (
    coerce_state_records as _coerce_state_records,
)
from .._multi_step_loop_state import (
    coerce_string_list as _coerce_string_list,
)
from .._multi_step_loop_state import (
    coerce_tool_results as _coerce_tool_results,
)
from .._multi_step_memory import (
    retrieve_memory_context,
    write_memory_observation,
)
from .._prompt_alternatives import (
    AlternativesPromptTarget,
    normalize_alternatives_prompt_target,
)
from .._prompt_overrides import (
    resolve_prompt_text,
)
from .._workflow_loop_orchestration import (
    compile_workflow_loop,
)


class MultiStepJsonToolCallingAgent(Delegate):
    """Agent that iterates explicit JSON action steps until completion."""

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
            llm_client: LLM client used for step generation.
            tool_runtime: Tool runtime shared across all steps.
            max_steps: Maximum number of action-observation iterations.
            stop_on_step_failure: Whether to stop immediately when one step fails.
            continuation_system_prompt: Unused in JSON mode; preserved for signature stability.
            continuation_user_prompt_template: Unused in JSON mode; preserved for signature stability.
            step_user_prompt_template: Optional step user prompt template.
            tool_calling_system_prompt: Optional system prompt for tool selection step.
            tool_calling_user_prompt_template: Optional user template for tool selection step.
            alternatives_prompt_target: Prompt target for alternatives blocks.
            continuation_memory_tail_items: Unused in JSON mode; preserved for signature stability.
            step_memory_tail_items: Memory tail size for step prompts.
            memory_store: Optional persistent memory store for retrieval/write-back.
            memory_namespace: Namespace partition used for memory reads/writes.
            memory_read_top_k: Number of memory matches retrieved per step.
            memory_write_observations: Whether to persist per-step observations.
            allowed_tools: Optional tool allowlist used by action steps.
            tracer: Optional explicit tracer dependency.

        Raises:
            ValueError: If ``max_steps``, ``step_memory_tail_items``, or ``memory_read_top_k``
                are less than ``1``.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
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
        _ = continuation_system_prompt, continuation_user_prompt_template, continuation_memory_tail_items
        self._step_user_prompt_template = resolve_prompt_text(
            override=step_user_prompt_template,
            default_prompt_name="multi_step_json_step_user",
            field_name="step_user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(alternatives_prompt_target)
        self._tool_calling_system_prompt = tool_calling_system_prompt
        self._tool_calling_user_prompt_template = tool_calling_user_prompt_template
        self._step_memory_tail_items = step_memory_tail_items
        self._memory_store = memory_store
        self._memory_namespace = memory_namespace.strip() or "default"
        self._memory_read_top_k = memory_read_top_k
        self._memory_write_observations = memory_write_observations
        self._allowed_tools = tuple(allowed_tools) if allowed_tools is not None else None

    def run(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """The multi-step action-observation loop and return aggregated results."""
        return self.compile(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def compile(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile the multi-step action-observation loop into one loop workflow."""
        execution_context = resolve_agent_execution_context(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
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
        alternatives_prompt_target = self._alternatives_prompt_target

        step_agent = JsonActionStepRunner(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            system_prompt=self._tool_calling_system_prompt,
            user_prompt_template=self._tool_calling_user_prompt_template,
            alternatives_prompt_target=alternatives_prompt_target,
            allowed_tools=self._allowed_tools,
            tracer=self._tracer,
        )

        workflow = compile_workflow_loop(
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
                step_agent=step_agent,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                stop_on_step_failure=stop_on_step_failure,
            ),
            tracer=self._tracer,
        )
        self.workflow = workflow

        def _finalize(workflow_result: ExecutionResult) -> ExecutionResult:
            loop_step_result = workflow_result.step_results.get("agent_loop")
            loop_output = loop_step_result.output if loop_step_result is not None else {}
            final_state_raw = loop_output.get("final_state", {})
            final_state = dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}
            result = build_json_final_result(
                final_state=final_state,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                max_steps=max_steps,
                stop_on_step_failure=stop_on_step_failure,
                alternatives_prompt_target=alternatives_prompt_target,
                step_memory_tail_items=self._step_memory_tail_items,
                memory_namespace=self._memory_namespace,
                memory_read_top_k=self._memory_read_top_k,
                memory_write_observations=self._memory_write_observations,
                memory_store_enabled=self._memory_store is not None,
            )
            merged_output = dict(result.output)
            merged_output["workflow"] = workflow_result.to_dict()
            merged_output["artifacts"] = workflow_result.output.get("artifacts", [])
            return ExecutionResult(
                output=merged_output,
                success=result.success,
                tool_results=list(result.tool_results),
                model_response=result.model_response,
                metadata=dict(result.metadata),
                step_results=dict(result.step_results),
                execution_order=list(result.execution_order),
            )

        return CompiledExecution(
            workflow=workflow,
            input={},
            request_id=resolved_request_id,
            workflow_request_id=f"{resolved_request_id}:workflow_loop",
            dependencies=resolved_dependencies,
            delegate_name="MultiStepJsonToolCallingAgent",
            tracer=self._tracer,
            trace_input=execution_context.normalized_input,
            finalize=_finalize,
        )

    def _run_loop_iteration(
        self,
        *,
        iteration: int,
        state: Mapping[str, object],
        prompt: str,
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
            step_agent: Step-level JSON tool agent instance.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency payload mapping.
            stop_on_step_failure: Effective stop-on-failure setting.

        Returns:
            Next loop-state mapping.
        """
        step_number = iteration
        memory = _coerce_state_records(state.get("memory"))
        decision_trace = _coerce_state_records(state.get("decision_trace"))
        retrieval_trace = _coerce_state_records(state.get("retrieval_trace"))
        memory_errors = _coerce_string_list(state.get("memory_errors"))
        step_outputs = _coerce_state_records(state.get("step_outputs"))
        tool_results = _coerce_tool_results(state.get("tool_results"))
        final_output = _coerce_mapping(state.get("final_output"))
        maybe_model_response = state.get("last_model_response")
        last_model_response = maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None

        retrieved_context, retrieved_matches, retrieval_error = retrieve_memory_context(
            memory_store=self._memory_store,
            namespace=self._memory_namespace,
            top_k=self._memory_read_top_k,
            task_prompt=prompt,
            memory=memory,
            memory_tail_items=self._step_memory_tail_items,
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
        step_final_output = normalize_step_final_output(step_result.output.get("final_output"))
        step_error = resolve_step_error(step_result)
        step_action_type = str(step_result.output.get("action_type", "tool_call"))
        final_answer_called = bool(step_result.output.get("final_answer_called", False))
        step_reason = str(step_result.output.get("reason", "")).strip()
        action_name = str(step_result.output.get("tool_name", "") or "")
        decision_record = {
            "step": step_number,
            "action_type": step_action_type,
            "action_name": action_name,
            "final_answer_called": final_answer_called,
            "success": step_result.success,
            "reason": step_reason,
        }
        decision_trace.append(decision_record)
        step_outputs.append(
            {
                "step": step_number,
                "success": step_result.success,
                "action_type": step_action_type,
                "final_answer_called": final_answer_called,
                "reason": step_reason,
                "final_output": step_final_output,
                "tool_name": step_result.output.get("tool_name"),
                "tool_input": step_result.output.get("tool_input", {}),
                "error": step_error,
                "tool_results_count": len(step_result.tool_results),
            }
        )
        if step_reason:
            memory.append(
                {
                    "kind": "thought",
                    "step": step_number,
                    "text": step_reason,
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
                    "thought": step_reason,
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
        should_continue_next = not final_answer_called
        fatal_error: str | None = None
        fatal_metadata: dict[str, object] = {}
        if step_result.success:
            if final_answer_called:
                final_output = step_final_output
                terminated_reason = TERMINATED_COMPLETED
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
            "decision_trace": decision_trace,
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
