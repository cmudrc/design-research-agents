"""Multi-step code tool-calling agent implemented via workflow loop primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._llm import LLMClient, LLMResponse
from design_research_agents._contracts._memory import MemoryStore
from design_research_agents._contracts._tools import ToolRuntime
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution

from .._code_action_step_runner import (
    CodeActionStepRunner,
)
from .._execution_context import (
    resolve_agent_execution_context,
)
from .._model_resolution import (
    resolve_agent_model,
)
from .._multi_step_code_runtime_helpers import (
    MultiStepCodeRunConfig,
    build_code_final_result,
)
from .._multi_step_code_runtime_helpers import (
    append_retrieval_trace as _append_retrieval_trace,
)
from .._multi_step_code_runtime_helpers import (
    build_code_step_agent as _build_code_step_agent,
)
from .._multi_step_code_runtime_helpers import (
    build_continuation_stop_state as _build_continuation_stop_state,
)
from .._multi_step_code_runtime_helpers import (
    build_step_input as _build_step_input,
)
from .._multi_step_code_runtime_helpers import (
    build_step_tools_text as _build_step_tools_text,
)
from .._multi_step_code_runtime_helpers import (
    resolve_run_config as _resolve_run_config,
)
from .._multi_step_code_runtime_helpers import (
    resolve_step_completion as _resolve_step_completion,
)
from .._multi_step_code_runtime_helpers import (
    write_step_observation as _write_step_observation,
)
from .._multi_step_common import (
    build_step_prompt,
)
from .._multi_step_continuation import (
    llm_should_continue as _llm_should_continue,
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
)
from .._prompt_alternatives import (
    AlternativesPromptTarget,
    normalize_alternatives_prompt_target,
)
from .._prompt_overrides import (
    resolve_prompt_text,
)
from .._response_schemas import (
    build_continuation_response_schema,
)
from .._workflow_loop_orchestration import (
    compile_workflow_loop,
)


class MultiStepCodeToolCallingAgent(Delegate):
    """Iterative code-tool agent with continuation checks between action steps."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        max_steps: int = 5,
        max_tool_calls_per_step: int = 5,
        execution_timeout_seconds: int = 5,
        validate_tool_input_schema: bool = False,
        normalize_generated_code_per_step: bool = False,
        stop_on_step_failure: bool = True,
        default_tools_per_step: Sequence[Mapping[str, object]] | None = None,
        continuation_system_prompt: str | None = None,
        continuation_user_prompt_template: str | None = None,
        step_user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        continuation_memory_tail_items: int = 6,
        step_memory_tail_items: int = 8,
        memory_store: MemoryStore | None = None,
        memory_namespace: str = "default",
        memory_read_top_k: int = 4,
        memory_write_observations: bool = True,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a multi-step code-tool agent.

        Args:
            llm_client: LLM client used for continuation and code-generation steps.
            tool_runtime: Tool runtime shared across all steps.
            max_steps: Maximum number of action-observation iterations.
            max_tool_calls_per_step: Per-step cap on tool calls.
            execution_timeout_seconds: Per-step execution timeout passed to the step agent.
            validate_tool_input_schema: Whether to validate tool inputs before invocation.
            normalize_generated_code_per_step: Whether to normalize generated code before run.
            stop_on_step_failure: Whether to stop immediately on step failures.
            default_tools_per_step: Optional default tool allowlist per step.
            continuation_system_prompt: Optional continuation system prompt override.
            continuation_user_prompt_template: Optional continuation user template override.
            step_user_prompt_template: Optional step user template override.
            alternatives_prompt_target: Prompt target for alternatives blocks.
            continuation_memory_tail_items: Memory tail size for continuation prompts.
            step_memory_tail_items: Memory tail size for step prompts.
            memory_store: Optional persistent memory store.
            memory_namespace: Namespace partition for memory reads/writes.
            memory_read_top_k: Number of memory matches retrieved per step.
            memory_write_observations: Whether to persist per-step observation summaries.
            tracer: Optional explicit tracer dependency.

        Raises:
            ValueError: Raised when numeric constructor bounds are invalid.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if max_tool_calls_per_step < 1:
            raise ValueError("max_tool_calls_per_step must be >= 1.")
        if execution_timeout_seconds < 1:
            raise ValueError("execution_timeout_seconds must be >= 1.")
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
        self._max_tool_calls_per_step = max_tool_calls_per_step
        self._execution_timeout_seconds = execution_timeout_seconds
        self._validate_tool_input_schema = validate_tool_input_schema
        self._normalize_generated_code_per_step = normalize_generated_code_per_step
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
            default_prompt_name="multi_step_step_user",
            field_name="step_user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(alternatives_prompt_target)
        self._continuation_memory_tail_items = continuation_memory_tail_items
        self._step_memory_tail_items = step_memory_tail_items
        self._memory_store = memory_store
        self._memory_namespace = memory_namespace.strip() or "default"
        self._memory_read_top_k = memory_read_top_k
        self._memory_write_observations = memory_write_observations
        self._default_tools_per_step = (
            tuple(dict(default_tool) for default_tool in default_tools_per_step if isinstance(default_tool, Mapping))
            if default_tools_per_step is not None
            else None
        )
        self._continuation_response_schema = build_continuation_response_schema()

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """The multi-step loop and return aggregated code-agent output."""
        return self.compile(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def compile(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile the multi-step code loop into one loop workflow."""
        execution_context = resolve_agent_execution_context(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        resolved_request_id = execution_context.request_id
        resolved_dependencies = execution_context.dependencies
        prompt = execution_context.prompt

        run_config = _resolve_run_config(
            input_payload=execution_context.normalized_input,
            defaults=MultiStepCodeRunConfig(
                max_steps=self._max_steps,
                max_tool_calls_per_step=self._max_tool_calls_per_step,
                execution_timeout_seconds=self._execution_timeout_seconds,
                validate_tool_input_schema=self._validate_tool_input_schema,
                stop_on_step_failure=self._stop_on_step_failure,
            ),
        )
        max_steps = run_config.max_steps
        max_tool_calls_per_step = run_config.max_tool_calls_per_step
        execution_timeout_seconds = run_config.execution_timeout_seconds
        validate_tool_input_schema = run_config.validate_tool_input_schema
        stop_on_step_failure = run_config.stop_on_step_failure
        normalize_generated_code_per_step = self._normalize_generated_code_per_step
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        alternatives_prompt_target = self._alternatives_prompt_target
        step_tools_text = _build_step_tools_text(
            tool_specs={spec.name: spec for spec in self._tool_runtime.list_tools()},
            default_tools_per_step=self._default_tools_per_step,
        )

        step_agent = _build_code_step_agent(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            max_tool_calls_per_step=max_tool_calls_per_step,
            execution_timeout_seconds=execution_timeout_seconds,
            validate_tool_input_schema=validate_tool_input_schema,
            normalize_generated_code_per_step=normalize_generated_code_per_step,
            default_tools_per_step=self._default_tools_per_step,
            alternatives_prompt_target=alternatives_prompt_target,
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
                max_steps=max_steps,
                resolved_model=resolved_model,
                alternatives_prompt_target=alternatives_prompt_target,
                step_tools_text=step_tools_text,
                step_agent=step_agent,
                normalized_input=execution_context.normalized_input,
                max_tool_calls_per_step=max_tool_calls_per_step,
                execution_timeout_seconds=execution_timeout_seconds,
                validate_tool_input_schema=validate_tool_input_schema,
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
            result = build_code_final_result(
                final_state=final_state,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                max_steps=max_steps,
                max_tool_calls_per_step=max_tool_calls_per_step,
                execution_timeout_seconds=execution_timeout_seconds,
                validate_tool_input_schema=validate_tool_input_schema,
                normalize_generated_code_per_step=normalize_generated_code_per_step,
                stop_on_step_failure=stop_on_step_failure,
                alternatives_prompt_target=alternatives_prompt_target,
                continuation_memory_tail_items=self._continuation_memory_tail_items,
                step_memory_tail_items=self._step_memory_tail_items,
                memory_namespace=self._memory_namespace,
                memory_read_top_k=self._memory_read_top_k,
                memory_write_observations=self._memory_write_observations,
                default_tools_per_step=self._default_tools_per_step,
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
            delegate_name="MultiStepCodeToolCallingAgent",
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
        max_steps: int,
        resolved_model: str,
        alternatives_prompt_target: AlternativesPromptTarget,
        step_tools_text: str,
        step_agent: CodeActionStepRunner,
        normalized_input: Mapping[str, object],
        max_tool_calls_per_step: int,
        execution_timeout_seconds: int,
        validate_tool_input_schema: bool,
        request_id: str,
        dependencies: Mapping[str, object],
        stop_on_step_failure: bool,
    ) -> Mapping[str, object]:
        """Execute one loop iteration and produce the next loop state.

        Args:
            iteration: One-based loop iteration number.
            state: Current loop-state mapping.
            prompt: User prompt text.
            max_steps: Effective max-step limit.
            resolved_model: Resolved model identifier.
            alternatives_prompt_target: Prompt target for alternatives injection.
            step_tools_text: Alternatives/tool block text.
            step_agent: Step-level code tool agent instance.
            normalized_input: Normalized run input payload.
            max_tool_calls_per_step: Effective per-step tool call cap.
            execution_timeout_seconds: Effective per-step timeout.
            validate_tool_input_schema: Effective tool input validation flag.
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
        last_model_response = maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None

        retrieved_context, retrieved_matches, retrieval_error = retrieve_memory_context(
            memory_store=self._memory_store,
            namespace=self._memory_namespace,
            top_k=self._memory_read_top_k,
            task_prompt=prompt,
            memory=memory,
            memory_tail_items=self._continuation_memory_tail_items,
        )
        _append_retrieval_trace(
            retrieval_trace=retrieval_trace,
            memory_errors=memory_errors,
            retrieval_error=retrieval_error,
            step_number=step_number,
            namespace=self._memory_namespace,
            retrieved_match_count=len(retrieved_matches),
        )

        should_continue, continue_reason, continue_source, continue_response = _llm_should_continue(
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
            alternatives_section_label="Allowed tools for action steps",
            agent_name="MultiStepCodeToolCallingAgent",
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
            return _build_continuation_stop_state(
                memory=memory,
                continuation_trace=continuation_trace,
                retrieval_trace=retrieval_trace,
                memory_errors=memory_errors,
                step_outputs=step_outputs,
                tool_results=tool_results,
                final_output=final_output,
                last_model_response=last_model_response,
                continue_source=continue_source,
            )

        step_prompt = build_step_prompt(
            prompt=prompt,
            memory=memory,
            step_number=step_number,
            prompt_template=self._step_user_prompt_template,
            memory_tail_items=self._step_memory_tail_items,
            retrieved_context=retrieved_context,
        )
        step_input = _build_step_input(
            normalized_input=normalized_input,
            step_prompt=step_prompt,
            max_tool_calls_per_step=max_tool_calls_per_step,
            execution_timeout_seconds=execution_timeout_seconds,
            validate_tool_input_schema=validate_tool_input_schema,
            alternatives_prompt_target=str(alternatives_prompt_target),
        )
        step_result = step_agent.run(
            json.dumps(step_input, indent=2, sort_keys=True, default=str),
            request_id=f"{request_id}:step-{step_number}",
            dependencies=dependencies,
        )
        if step_result.model_response is not None:
            last_model_response = step_result.model_response

        tool_results.extend(step_result.tool_results)
        step_outputs.append(
            {
                "step": step_number,
                "success": step_result.success,
                "final_output": step_result.output.get("final_output", {}),
                "error": step_result.output.get("error"),
                "tool_results_count": len(step_result.tool_results),
            }
        )
        memory.extend(
            [
                {
                    "kind": "action",
                    "step": step_number,
                    "generated_code": step_result.output.get("generated_code", ""),
                },
                {
                    "kind": "observation",
                    "step": step_number,
                    "success": step_result.success,
                    "final_output": step_result.output.get("final_output", {}),
                    "error": step_result.output.get("error"),
                },
            ]
        )

        if self._memory_write_observations:
            memory_write_error = _write_step_observation(
                memory_store=self._memory_store,
                namespace=self._memory_namespace,
                task_prompt=prompt,
                step_number=step_number,
                continuation_reason=continue_reason,
                step_success=step_result.success,
                generated_code=step_result.output.get("generated_code", ""),
                final_output=step_result.output.get("final_output"),
                error=step_result.output.get("error"),
            )
            if memory_write_error is not None:
                memory_errors.append(f"write(step {step_number}): {memory_write_error}")

        final_output, terminated_reason, should_continue_next, fatal_error, fatal_metadata = _resolve_step_completion(
            step_result=step_result,
            step_outputs=step_outputs,
            memory=memory,
            final_output=final_output,
            stop_on_step_failure=stop_on_step_failure,
        )
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
    "MultiStepCodeToolCallingAgent",
]
