"""Multi-step ReAct-style tool router with TOOL_CALL/STOP controller steps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.termination import (
    TERMINATED_INVALID_STEP_OUTPUT,
    stop_reason,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime
from design_research_agents.tracing import (
    Tracer,
    emit_continuation_decision,
    finish_model_call,
    start_model_call,
)

from ..shared.agent_internal.execution_context import (
    finish_agent_execution,
    prepare_agent_execution,
)
from ..shared.agent_internal.input_parsing import (
    extract_boolean as _extract_boolean,
)
from ..shared.agent_internal.input_parsing import (
    extract_positive_int as _extract_positive_int,
)
from ..shared.agent_internal.model_resolution import resolve_agent_model
from ..shared.agent_internal.multi_step_common import (
    build_step_prompt,
)
from ..shared.agent_internal.multi_step_loop_state import (
    build_loop_initial_state,
    continue_loop,
)
from ..shared.agent_internal.multi_step_loop_state import (
    coerce_state_records as _coerce_state_records,
)
from ..shared.agent_internal.multi_step_loop_state import (
    coerce_tool_results as _coerce_tool_results,
)
from ..shared.agent_internal.multi_step_router_runtime_helpers import (
    build_router_final_result,
)
from ..shared.agent_internal.multi_step_router_runtime_helpers import (
    run_tool_call_step as _run_tool_call_step,
)
from ..shared.agent_internal.multi_step_tool_router_helpers import (
    ToolRouterStepDecision,
)
from ..shared.agent_internal.multi_step_tool_router_helpers import (
    normalize_output_dict as _normalize_output_dict,
)
from ..shared.agent_internal.multi_step_tool_router_helpers import (
    parse_tool_router_step_decision as _parse_tool_router_step_decision,
)
from ..shared.agent_internal.prompt_alternatives import (
    AlternativesPromptTarget,
    inject_alternatives_into_prompt_pair,
    normalize_alternatives_prompt_target,
)
from ..shared.agent_internal.prompt_overrides import (
    resolve_prompt_text,
)
from ..shared.agent_internal.response_schemas import (
    build_multi_step_tool_router_response_schema,
    clone_response_schema,
)
from ..shared.agent_internal.router_agent_helpers import (
    ToolAlternative,
    build_routes_text,
    compile_runtime_alternatives,
    extract_alternatives,
    resolve_allowed_route_names,
)
from ..shared.agent_internal.workflow_loop_orchestration import run_workflow_loop


class MultiStepToolRouterAgent(Agent):
    """Agent that iterates tool-routing steps until a STOP decision."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        max_steps: int = 5,
        stop_on_step_failure: bool = True,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        allowed_routes: Sequence[str] | None = None,
        step_memory_tail_items: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a multi-step tool router.

        Args:
            llm_client: LLM client used for each step decision.
            tool_runtime: Tool runtime shared across steps.
            max_steps: Maximum number of decision steps.
            stop_on_step_failure: Whether to stop immediately after one failed tool call.
            system_prompt: Optional step controller system prompt override.
            user_prompt_template: Optional step controller user prompt template override.
            alternatives_prompt_target: Prompt target for alternatives blocks.
            allowed_routes: Optional route/tool allowlist.
            step_memory_tail_items: Memory tail size for step prompts.
            tracer: Optional explicit tracer dependency.

        Raises:
            ValueError: Raised when constructor limits or route filters are invalid.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if step_memory_tail_items < 1:
            raise ValueError("step_memory_tail_items must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self.workflow: object | None = None
        self._max_steps = max_steps
        self._stop_on_step_failure = stop_on_step_failure
        self._system_prompt = resolve_prompt_text(
            override=system_prompt,
            default_prompt_name="multi_step_tool_router_system",
            field_name="system_prompt",
        )
        self._user_prompt_template = resolve_prompt_text(
            override=user_prompt_template,
            default_prompt_name="multi_step_tool_router_user",
            field_name="user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(
            alternatives_prompt_target
        )
        self._step_memory_tail_items = step_memory_tail_items
        self._runtime_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        self._allowed_route_names = resolve_allowed_route_names(
            runtime_specs=self._runtime_specs,
            allowed_routes=allowed_routes,
        )
        self._compiled_runtime_alternatives = compile_runtime_alternatives(
            tool_specs=self._runtime_specs,
            allowed_route_names=self._allowed_route_names,
        )
        self._default_alternatives = extract_alternatives(
            runtime_specs=self._runtime_specs,
            compiled_runtime_alternatives=self._compiled_runtime_alternatives,
        )
        self._step_response_schema = build_multi_step_tool_router_response_schema(
            tool_names=[alternative.tool_name for alternative in self._default_alternatives]
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run a multi-step TOOL_CALL/STOP loop and return aggregated output.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional request id for tracing and correlation.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final normalized execution result for the run.

        Raises:
            Exception: Propagates execution failures from nested workflow/LLM/tool calls.
        """
        execution_context = prepare_agent_execution(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            agent_name="MultiStepToolRouterAgent",
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
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        alternatives: list[ToolAlternative] = [
            dict_alternative for dict_alternative in self._default_alternatives
        ]
        routes_text = build_routes_text(alternatives=alternatives)

        try:
            loop_result = run_workflow_loop(
                max_iterations=max_steps,
                initial_state=build_loop_initial_state(
                    prompt=prompt,
                    include_continuation=False,
                ),
                continue_predicate=continue_loop,
                iteration_handler=lambda iteration, state: self._run_loop_iteration(
                    iteration=iteration,
                    state=state,
                    prompt=prompt,
                    resolved_model=resolved_model,
                    routes_text=routes_text,
                    alternatives=alternatives,
                    normalized_input=execution_context.normalized_input,
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
        result = build_router_final_result(
            final_state=loop_result.final_state,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
            max_steps=max_steps,
            stop_on_step_failure=stop_on_step_failure,
            alternatives_prompt_target=str(self._alternatives_prompt_target),
            step_memory_tail_items=self._step_memory_tail_items,
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
        resolved_model: str,
        routes_text: str,
        alternatives: Sequence[ToolAlternative],
        normalized_input: Mapping[str, object],
        request_id: str,
        dependencies: Mapping[str, object],
        stop_on_step_failure: bool,
    ) -> Mapping[str, object]:
        """Execute one router loop iteration and produce next loop state.

        Args:
            iteration: One-based loop iteration number.
            state: Current loop-state mapping.
            prompt: User prompt text.
            resolved_model: Resolved model identifier.
            routes_text: Rendered route alternatives text block.
            alternatives: Normalized routing alternatives.
            normalized_input: Normalized run input payload.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency payload mapping.
            stop_on_step_failure: Effective stop-on-failure setting.

        Returns:
            Next loop-state mapping.

        Raises:
            Exception: Propagates model call failures.
        """
        step_number = iteration
        memory = _coerce_state_records(state.get("memory"))
        step_outputs = _coerce_state_records(state.get("step_outputs"))
        tool_results = _coerce_tool_results(state.get("tool_results"))
        final_output = _normalize_output_dict(state.get("final_output"))
        maybe_model_response = state.get("last_model_response")
        last_model_response = (
            maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None
        )

        user_prompt = build_step_prompt(
            prompt=prompt,
            memory=memory,
            step_number=step_number,
            prompt_template=self._user_prompt_template,
            memory_tail_items=self._step_memory_tail_items,
        )
        system_prompt, user_prompt = inject_alternatives_into_prompt_pair(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            section_label="Available routes",
            alternatives_text=routes_text,
            target=self._alternatives_prompt_target,
        )
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        llm_params = LLMChatParams(
            response_schema=clone_response_schema(self._step_response_schema),
            provider_options={
                "agent": "MultiStepToolRouterAgent",
                "phase": "step_controller",
            },
        )
        model_span_id = start_model_call(
            model=resolved_model,
            messages=messages,
            params=llm_params,
            metadata={
                "agent": "MultiStepToolRouterAgent",
                "phase": "step_controller",
            },
        )
        try:
            llm_response = self._llm_client.chat(
                messages,
                model=resolved_model,
                params=llm_params,
            )
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            raise
        finish_model_call(model_span_id, response=llm_response)
        last_model_response = llm_response

        parsed_step = _parse_tool_router_step_decision(llm_response.text)
        if parsed_step is None:
            return {
                "memory": memory,
                "step_outputs": step_outputs,
                "tool_results": tool_results,
                "final_output": final_output,
                "last_model_response": last_model_response,
                "terminated_reason": TERMINATED_INVALID_STEP_OUTPUT,
                "should_continue": False,
                "fatal_error": (
                    "Multi-step tool router step output was invalid. "
                    "Expected TOOL_CALL or STOP JSON."
                ),
                "fatal_metadata": {
                    "stage": "step_decision",
                    "terminated_reason": TERMINATED_INVALID_STEP_OUTPUT,
                },
            }

        if parsed_step.action == "STOP":
            final_output = _normalize_output_dict(parsed_step.final_output) or final_output
            terminated_reason = stop_reason(parsed_step.source)
            emit_continuation_decision(
                step=step_number,
                should_continue=False,
                reason=parsed_step.reason,
                source=parsed_step.source,
            )
            step_outputs.append(
                {
                    "step": step_number,
                    "action": "STOP",
                    "reason": parsed_step.reason,
                    "source": parsed_step.source,
                    "final_output": final_output,
                    "success": True,
                }
            )
            memory.append(
                {
                    "kind": "stop",
                    "step": step_number,
                    "final_output": final_output,
                    "reason": parsed_step.reason,
                    "source": parsed_step.source,
                }
            )
            return {
                "memory": memory,
                "step_outputs": step_outputs,
                "tool_results": tool_results,
                "final_output": final_output,
                "last_model_response": last_model_response,
                "terminated_reason": terminated_reason,
                "should_continue": False,
                "fatal_error": None,
                "fatal_metadata": {},
            }

        return self._run_tool_call_step(
            step_number=step_number,
            parsed_step=parsed_step,
            alternatives=alternatives,
            normalized_input=normalized_input,
            request_id=request_id,
            dependencies=dependencies,
            memory=memory,
            step_outputs=step_outputs,
            tool_results=tool_results,
            final_output=final_output,
            last_model_response=last_model_response,
            stop_on_step_failure=stop_on_step_failure,
        )

    def _run_tool_call_step(
        self,
        *,
        step_number: int,
        parsed_step: ToolRouterStepDecision,
        alternatives: Sequence[ToolAlternative],
        normalized_input: Mapping[str, object],
        request_id: str,
        dependencies: Mapping[str, object],
        memory: list[dict[str, object]],
        step_outputs: list[dict[str, object]],
        tool_results: list[ToolResult],
        final_output: dict[str, object],
        last_model_response: LLMResponse | None,
        stop_on_step_failure: bool,
    ) -> Mapping[str, object]:
        """Delegate TOOL_CALL step handling to shared runtime helpers.

        Args:
            step_number: One-based step number.
            parsed_step: Parsed router controller decision.
            alternatives: Normalized routing alternatives.
            normalized_input: Normalized run input payload.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency payload mapping.
            memory: Mutable memory record list.
            step_outputs: Mutable step output list.
            tool_results: Mutable tool result list.
            final_output: Current run-level final output mapping.
            last_model_response: Most recent model response.
            stop_on_step_failure: Effective stop-on-failure setting.

        Returns:
            Next loop-state mapping.
        """
        return _run_tool_call_step(
            tool_runtime=self._tool_runtime,
            step_number=step_number,
            parsed_step=parsed_step,
            alternatives=list(alternatives),
            normalized_input=normalized_input,
            request_id=request_id,
            dependencies=dependencies,
            memory=memory,
            step_outputs=step_outputs,
            tool_results=tool_results,
            final_output=final_output,
            last_model_response=last_model_response,
            stop_on_step_failure=stop_on_step_failure,
        )
