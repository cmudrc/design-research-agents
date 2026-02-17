"""Multi-step ReAct-style tool router with TOOL_CALL/STOP controller steps."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from design_research_agents.agent.internal.input_parsing import (
    extract_boolean as _extract_boolean,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_positive_int as _extract_positive_int,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.multi_step_common import build_step_prompt
from design_research_agents.agent.internal.multi_step_tool_router_helpers import (
    failure_result as _failure_result,
)
from design_research_agents.agent.internal.multi_step_tool_router_helpers import (
    normalize_output_dict as _normalize_output_dict,
)
from design_research_agents.agent.internal.multi_step_tool_router_helpers import (
    parse_tool_router_step_decision as _parse_tool_router_step_decision,
)
from design_research_agents.agent.internal.multi_step_tool_router_helpers import (
    resolve_selected_tool as _resolve_selected_tool,
)
from design_research_agents.agent.internal.prompt_alternatives import (
    AlternativesPromptTarget,
    inject_alternatives_into_prompt_pair,
    normalize_alternatives_prompt_target,
)
from design_research_agents.agent.internal.prompt_overrides import resolve_prompt_text
from design_research_agents.agent.internal.response_schemas import (
    build_multi_step_tool_router_response_schema,
    clone_response_schema,
)
from design_research_agents.agent.internal.router_agent_helpers import (
    build_routes_text,
    compile_runtime_alternatives,
    extract_alternatives,
    resolve_allowed_route_names,
    resolve_tool_input,
)
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime
from design_research_agents.tracing import (
    Tracer,
    emit_continuation_decision,
    emit_guardrail_decision,
    emit_router_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


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
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if step_memory_tail_items < 1:
            raise ValueError("step_memory_tail_items must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
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
    ) -> AgentResult:
        """Run a multi-step TOOL_CALL/STOP loop and return aggregated output."""
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        trace_scope = start_trace_run(
            agent_name="MultiStepToolRouterAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )
        prompt = _extract_prompt(normalized_input)
        max_steps = _extract_positive_int(
            input_payload=normalized_input,
            key="max_steps",
            default_value=self._max_steps,
        )
        stop_on_step_failure = _extract_boolean(
            input_payload=normalized_input,
            key="stop_on_step_failure",
            default_value=self._stop_on_step_failure,
        )
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        alternatives = [dict_alternative for dict_alternative in self._default_alternatives]
        routes_text = build_routes_text(alternatives=alternatives)

        memory: list[dict[str, object]] = [{"kind": "task", "prompt": prompt}]
        step_outputs: list[dict[str, object]] = []
        tool_results: list[ToolResult] = []
        final_output: dict[str, object] = {}
        last_model_response: LLMResponse | None = None
        terminated_reason = "max_steps_reached"

        for step_index in range(max_steps):
            step_number = step_index + 1
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
                finish_trace_run(trace_scope, error=str(exc))
                raise
            finish_model_call(model_span_id, response=llm_response)
            last_model_response = llm_response

            parsed_step = _parse_tool_router_step_decision(llm_response.text)
            if parsed_step is None:
                result = _failure_result(
                    error=(
                        "Multi-step tool router step output was invalid. "
                        "Expected TOOL_CALL or STOP JSON."
                    ),
                    model_response=last_model_response,
                    tool_results=tool_results,
                    request_id=resolved_request_id,
                    dependencies=resolved_dependencies,
                    metadata={
                        "stage": "step_decision",
                        "terminated_reason": "invalid_step_output",
                    },
                    output={
                        "final_output": final_output,
                        "steps_executed": len(step_outputs),
                        "step_outputs": step_outputs,
                        "memory": memory,
                        "terminated_reason": "invalid_step_output",
                    },
                )
                finish_trace_run(trace_scope, result=result)
                return result

            if parsed_step.action == "STOP":
                final_output = parsed_step.final_output or final_output
                terminated_reason = f"stop:{parsed_step.source}"
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
                break

            tool_resolution = _resolve_selected_tool(
                alternatives=alternatives,
                tool_names=parsed_step.tool_names,
            )
            if tool_resolution is None:
                emit_guardrail_decision(
                    guardrail="route_validation",
                    decision="reject",
                    reason="invalid selected tool_names",
                    details={"step": step_number},
                )
                result = _failure_result(
                    error="Step selected no valid tool route.",
                    model_response=last_model_response,
                    tool_results=tool_results,
                    request_id=resolved_request_id,
                    dependencies=resolved_dependencies,
                    metadata={
                        "stage": "step_decision",
                        "terminated_reason": "invalid_route_selection",
                    },
                    output={
                        "final_output": final_output,
                        "steps_executed": len(step_outputs),
                        "step_outputs": step_outputs,
                        "memory": memory,
                        "terminated_reason": "invalid_route_selection",
                    },
                )
                finish_trace_run(trace_scope, result=result)
                return result
            selected_tool_name, selected_tool_index = tool_resolution
            emit_router_decision(
                source=parsed_step.source,
                alternatives=[alternative.tool_name for alternative in alternatives],
                selected_tool_name=selected_tool_name,
                selected_index=selected_tool_index,
                reason=parsed_step.reason,
                parsed_route={
                    "action": "TOOL_CALL",
                    "tool_names": list(parsed_step.tool_names),
                    "reason": parsed_step.reason,
                },
            )

            tool_input = (
                parsed_step.tool_input
                if parsed_step.tool_input is not None
                else resolve_tool_input(
                    tool_name=selected_tool_name,
                    input_payload=normalized_input,
                )
            )
            tool_result = self._tool_runtime.invoke(
                selected_tool_name,
                tool_input,
                request_id=f"{resolved_request_id}:step-{step_number}",
                dependencies=resolved_dependencies,
            )
            tool_results.append(tool_result)
            emit_continuation_decision(
                step=step_number,
                should_continue=True,
                reason=parsed_step.reason,
                source=parsed_step.source,
            )

            step_final_output = _normalize_output_dict(tool_result.result)
            if tool_result.ok:
                final_output = step_final_output
            step_outputs.append(
                {
                    "step": step_number,
                    "action": "TOOL_CALL",
                    "tool_name": selected_tool_name,
                    "tool_names": list(parsed_step.tool_names),
                    "tool_input": tool_input,
                    "tool_output": tool_result.result,
                    "reason": parsed_step.reason,
                    "source": parsed_step.source,
                    "success": tool_result.ok,
                    "error": tool_result.error,
                }
            )
            memory.extend(
                [
                    {
                        "kind": "action",
                        "step": step_number,
                        "tool_name": selected_tool_name,
                        "tool_names": list(parsed_step.tool_names),
                        "tool_input": tool_input,
                    },
                    {
                        "kind": "observation",
                        "step": step_number,
                        "success": tool_result.ok,
                        "final_output": step_final_output,
                        "error": tool_result.error,
                    },
                ]
            )

            if tool_result.ok:
                continue
            terminated_reason = "step_failure"
            if stop_on_step_failure:
                result = _failure_result(
                    error=(
                        tool_result.error.message
                        if tool_result.error is not None
                        else "Step tool execution failed."
                    ),
                    model_response=last_model_response,
                    tool_results=tool_results,
                    request_id=resolved_request_id,
                    dependencies=resolved_dependencies,
                    metadata={
                        "stage": "step_execution",
                        "terminated_reason": terminated_reason,
                    },
                    output={
                        "final_output": final_output,
                        "steps_executed": len(step_outputs),
                        "step_outputs": step_outputs,
                        "memory": memory,
                        "terminated_reason": terminated_reason,
                    },
                )
                finish_trace_run(trace_scope, result=result)
                return result

        success = all(
            step_output.get("success") is True
            for step_output in step_outputs
            if step_output.get("action") == "TOOL_CALL"
        )
        result = AgentResult(
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
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "config": {
                    "max_steps": max_steps,
                    "stop_on_step_failure": stop_on_step_failure,
                    "alternatives_prompt_target": self._alternatives_prompt_target,
                    "step_memory_tail_items": self._step_memory_tail_items,
                },
            },
        )
        finish_trace_run(trace_scope, result=result)
        return result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Emit a deterministic stream wrapper around ``run``."""
        result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)
