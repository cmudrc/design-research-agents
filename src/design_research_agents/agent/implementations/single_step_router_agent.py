"""Router agent implementation that selects one tool alternative per request.

The router asks the model to choose a route from runtime-backed alternatives,
requires a structured route payload, executes the selected tool, and returns
both model and tool artifacts in a single ``AgentResult``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.prompt_alternatives import (
    AlternativesPromptTarget,
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    normalize_alternatives_prompt_target,
)
from design_research_agents.agent.internal.prompt_overrides import resolve_prompt_text
from design_research_agents.agent.internal.response_schemas import clone_response_schema
from design_research_agents.agent.internal.router_agent_helpers import (
    build_route_prompt,
    build_routes_text,
    clone_alternative,
    compile_runtime_alternatives,
    extract_alternatives,
    parse_route_response,
    resolve_allowed_route_names,
    resolve_model_route,
    resolve_tool_input,
    route_response_schema,
    routing_failure_result,
)
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.agent.internal.tool_input import extract_prompt
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
)
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.tracing import (
    Tracer,
    emit_guardrail_decision,
    emit_router_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


class SingleStepRouterAgent(Agent):
    """Agent that routes one request to one selected tool alternative.

    The agent compiles alternatives from tool runtime specs, prompts the model
    for a strict JSON route selection, and executes the selected tool only when
    the model output is valid.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        allowed_routes: Sequence[str] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a router agent with injected runtime dependencies.

        Args:
            llm_client: LLM client used for prompt execution.
            tool_runtime: Tool runtime used for tool invocation.
            system_prompt: Optional system prompt override.
            user_prompt_template: Optional user prompt template override.
            alternatives_prompt_target: Prompt target for routes block.
            allowed_routes: Optional route/tool allowlist.
            tracer: Optional explicit tracer dependency.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self._system_prompt = resolve_prompt_text(
            override=system_prompt,
            default_prompt_name="router_system",
            field_name="system_prompt",
        )
        self._user_prompt_template = resolve_prompt_text(
            override=user_prompt_template,
            default_prompt_name="router_user_route",
            field_name="user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(
            alternatives_prompt_target
        )
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
        self._default_route_response_schema = route_response_schema(
            alternatives=self._default_alternatives,
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run one model route-selection call and one routed tool invocation.

        Invalid model routing output is treated as a hard failure result instead
        of triggering deterministic routing fallbacks.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        trace_scope = start_trace_run(
            agent_name="SingleStepRouterAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )
        prompt = extract_prompt(normalized_input)
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        alternatives = [
            clone_alternative(alternative) for alternative in self._default_alternatives
        ]
        alternatives_prompt_target = self._alternatives_prompt_target
        routes_text = build_routes_text(alternatives=alternatives)
        routes_block = build_user_prompt_alternatives_block(
            section_label="Available routes",
            alternatives_text=routes_text,
            target=alternatives_prompt_target,
        )
        user_prompt = build_route_prompt(
            prompt=prompt,
            routes_block=routes_block,
            prompt_template=self._user_prompt_template,
        )
        system_prompt = self._system_prompt
        if alternatives_prompt_target == "system":
            system_prompt = append_alternatives_block(
                prompt_text=system_prompt,
                section_label="Available routes",
                alternatives_text=routes_text,
            )

        messages = [
            LLMMessage(
                role="system",
                content=system_prompt,
            ),
            LLMMessage(
                role="user",
                content=user_prompt,
            ),
        ]
        llm_params = LLMChatParams(
            response_schema=clone_response_schema(self._default_route_response_schema),
            provider_options={"agent": "SingleStepRouterAgent", "phase": "route_select"},
        )
        model_span_id = start_model_call(
            model=resolved_model,
            messages=messages,
            params=llm_params,
            metadata={"agent": "SingleStepRouterAgent", "phase": "route_select"},
        )
        try:
            llm_response = self._llm_client.chat(messages, model=resolved_model, params=llm_params)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            finish_trace_run(trace_scope, error=str(exc))
            raise
        finish_model_call(model_span_id, response=llm_response)
        parsed_route = parse_route_response(llm_response.text)

        route_resolution = resolve_model_route(
            parsed_route=parsed_route,
            alternatives=alternatives,
        )
        if route_resolution is None:
            emit_guardrail_decision(
                guardrail="route_validation",
                decision="reject",
                reason="invalid model route output",
                details={"stage": "routing"},
            )
            emit_router_decision(
                source="model_invalid",
                alternatives=[candidate.tool_name for candidate in alternatives],
                selected_tool_name=None,
                selected_index=None,
                reason="invalid model route output",
                parsed_route=(
                    {
                        "selection": parsed_route.selection,
                        "reason": parsed_route.reason,
                    }
                    if parsed_route is not None
                    else None
                ),
            )
            result = routing_failure_result(
                error=(
                    "Router model output was invalid. "
                    "Expected JSON with one valid discrete route `selection`."
                ),
                llm_response=llm_response,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                alternatives=alternatives,
                parsed_route=parsed_route,
            )
            finish_trace_run(trace_scope, result=result)
            return result
        selected_alternative, selected_index, selected_reason = route_resolution
        emit_router_decision(
            source="model",
            alternatives=[candidate.tool_name for candidate in alternatives],
            selected_tool_name=selected_alternative.tool_name,
            selected_index=selected_index,
            reason=selected_reason,
            parsed_route=(
                {
                    "selection": parsed_route.selection,
                    "reason": parsed_route.reason,
                }
                if parsed_route is not None
                else None
            ),
        )

        model_text = llm_response.text
        tool_input = resolve_tool_input(
            tool_name=selected_alternative.tool_name,
            input_payload=normalized_input,
        )
        tool_result = self._tool_runtime.invoke(
            selected_alternative.tool_name,
            tool_input,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
        )

        output: dict[str, object] = {
            "model_text": model_text,
            "model_response": {
                "selection": (parsed_route.selection if parsed_route is not None else None),
                "reason": parsed_route.reason if parsed_route is not None else None,
            },
            "tool_name": selected_alternative.tool_name,
            "selected_alternative_index": selected_index,
            "tool_input": tool_input,
            "tool_output": tool_result.result,
        }
        result = AgentResult(
            output=output,
            success=tool_result.ok,
            tool_results=[tool_result],
            model_response=llm_response,
            metadata={
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "routing": {
                    "source": "model",
                    "alternatives": [candidate.tool_name for candidate in alternatives],
                    "selected_tool_name": selected_alternative.tool_name,
                    "selected_alternative_index": selected_index,
                    "selected_reason": selected_reason,
                    "parsed_route": (
                        {
                            "selection": parsed_route.selection,
                            "reason": parsed_route.reason,
                        }
                        if parsed_route is not None
                        else None
                    ),
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
        """Emit a deterministic stream wrapper around ``run``.

        The wrapper currently emits one full-text delta event followed by a
        completion event that carries the full ``AgentResult`` payload.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Yields:
            Streaming events through completion.
        """
        result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)


__all__ = [
    "SingleStepRouterAgent",
]
