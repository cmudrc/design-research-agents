"""Workflow-native single-step tool router agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.llm import LLMChatParams, LLMClient, LLMMessage, LLMResponse
from design_research_agents.contracts.tools import ToolError, ToolResult, ToolRuntime
from design_research_agents.contracts.workflow import LogicStep, ToolStep, ToolStepInputBuilder
from design_research_agents.implementations.shared.agent_internal.execution_context import (
    finish_agent_execution,
    prepare_agent_execution,
)
from design_research_agents.implementations.shared.agent_internal.model_resolution import (
    resolve_agent_model,
)
from design_research_agents.implementations.shared.agent_internal.prompt_alternatives import (
    AlternativesPromptTarget,
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    normalize_alternatives_prompt_target,
)
from design_research_agents.implementations.shared.agent_internal.prompt_overrides import (
    resolve_prompt_text,
)
from design_research_agents.implementations.shared.agent_internal.response_schemas import (
    clone_response_schema,
)
from design_research_agents.implementations.shared.agent_internal.router_agent_helpers import (
    ParsedRoute,
    ToolAlternative,
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
from design_research_agents.implementations.shared.agent_internal.tool_input import extract_prompt
from design_research_agents.implementations.shared.agent_internal.workflow_first_envelope import (
    build_workflow_first_output,
)
from design_research_agents.tracing import (
    Tracer,
    emit_guardrail_decision,
    emit_router_decision,
    finish_model_call,
    start_model_call,
)
from design_research_agents.workflow import Workflow


class SingleStepToolRouterAgent(Agent):
    """Agent that routes one request to one selected tool alternative."""

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
        self._tool_step_ids = {
            alternative.tool_name: _tool_step_id(alternative.tool_name)
            for alternative in self._default_alternatives
        }
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run one model route-selection call and one routed tool invocation.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.

        Raises:
            Exception: Raised when execution fails.
        """
        execution_context = prepare_agent_execution(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            agent_name="SingleStepToolRouterAgent",
            tracer=self._tracer,
        )
        self.workflow = self._build_workflow()

        try:
            workflow_result = self.workflow.run(
                {
                    "normalized_input": execution_context.normalized_input,
                    "request_id": execution_context.request_id,
                },
                execution_mode="sequential",
                failure_policy="skip_dependents",
                request_id=f"{execution_context.request_id}:single_step_router",
                dependencies=execution_context.dependencies,
            )
            result = self._build_result(
                workflow_result=workflow_result,
                request_id=execution_context.request_id,
                dependencies=execution_context.dependencies,
            )
        except Exception as exc:
            finish_agent_execution(trace_scope=execution_context.trace_scope, error=str(exc))
            raise

        finish_agent_execution(trace_scope=execution_context.trace_scope, result=result)
        return result

    def _build_workflow(self) -> Workflow:
        """Compose route selection + per-route tool invocation workflow graph.

        Returns:
            Workflow configured for route selection, routed tool execution,
            and invalid-route handling.
        """
        route_map = {
            alternative.tool_name: (self._tool_step_ids[alternative.tool_name],)
            for alternative in self._default_alternatives
        }
        route_map["__invalid__"] = ("invalid_route",)

        steps: list[LogicStep | ToolStep] = [
            LogicStep(
                step_id="select_route",
                handler=self._select_route_handler,
                route_map=route_map,
            ),
        ]
        for alternative in self._default_alternatives:
            steps.append(
                ToolStep(
                    step_id=self._tool_step_ids[alternative.tool_name],
                    tool_name=alternative.tool_name,
                    dependencies=("select_route",),
                    input_builder=_build_tool_input_builder(
                        expected_tool_name=alternative.tool_name,
                    ),
                )
            )

        steps.append(
            LogicStep(
                step_id="invalid_route",
                dependencies=("select_route",),
                handler=_invalid_route_handler,
            )
        )

        return Workflow(
            tool_runtime=self._tool_runtime,
            tracer=self._tracer,
            input_mode="schema",
            steps=steps,
            default_execution_mode="sequential",
            default_failure_policy="skip_dependents",
        )

    def _select_route_handler(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Run model route-selection logic and emit a tool route key.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Mapping containing selected route metadata and model response payload.

        Raises:
            TypeError: If schema-mode input payloads are missing or malformed.
            Exception: Propagated when the model request fails.
        """
        inputs = context.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TypeError("Single-step router workflow requires schema input mapping.")
        normalized_input = inputs.get("normalized_input")
        request_id = inputs.get("request_id")
        if not isinstance(normalized_input, Mapping):
            raise TypeError("normalized_input must be a mapping.")

        resolved_request_id = str(request_id) if request_id is not None else ""
        prompt = extract_prompt(normalized_input)
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
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
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        llm_params = LLMChatParams(
            response_schema=clone_response_schema(self._default_route_response_schema),
            provider_options={
                "agent": "SingleStepToolRouterAgent",
                "phase": "route_select",
            },
        )
        model_span_id = start_model_call(
            model=resolved_model,
            messages=messages,
            params=llm_params,
            metadata={"agent": "SingleStepToolRouterAgent", "phase": "route_select"},
        )
        try:
            llm_response = self._llm_client.chat(messages, model=resolved_model, params=llm_params)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
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
                        "tool_names": list(parsed_route.tool_names),
                        "reason": parsed_route.reason,
                    }
                    if parsed_route is not None
                    else None
                ),
            )
            return {
                "route": "__invalid__",
                "selection_valid": False,
                "error": (
                    "Router model output was invalid. "
                    "Expected JSON with `tool_names` (non-empty list)."
                ),
                "model_text": llm_response.text,
                "model_response_payload": {},
                "selected_tool_name": None,
                "selected_tool_names": [],
                "selected_index": None,
                "selected_reason": "invalid model route output",
                "tool_input": {},
                "alternatives": [candidate.tool_name for candidate in alternatives],
                "parsed_route": (
                    {
                        "tool_names": list(parsed_route.tool_names),
                        "reason": parsed_route.reason,
                    }
                    if parsed_route is not None
                    else None
                ),
                "model_response": llm_response,
            }

        (
            selected_alternative,
            selected_index,
            selected_reason,
            selected_tool_names,
        ) = route_resolution
        emit_router_decision(
            source="model",
            alternatives=[candidate.tool_name for candidate in alternatives],
            selected_tool_name=selected_alternative.tool_name,
            selected_index=selected_index,
            reason=selected_reason,
            parsed_route=(
                {
                    "tool_names": list(parsed_route.tool_names),
                    "reason": parsed_route.reason,
                }
                if parsed_route is not None
                else None
            ),
        )

        tool_input = resolve_tool_input(
            tool_name=selected_alternative.tool_name,
            input_payload=normalized_input,
        )
        return {
            "route": selected_alternative.tool_name,
            "selection_valid": True,
            "error": None,
            "model_text": llm_response.text,
            "model_response_payload": {
                "tool_names": list(parsed_route.tool_names) if parsed_route is not None else [],
                "reason": parsed_route.reason if parsed_route is not None else None,
            },
            "selected_tool_name": selected_alternative.tool_name,
            "selected_tool_names": selected_tool_names,
            "selected_index": selected_index,
            "selected_reason": selected_reason,
            "tool_input": tool_input,
            "alternatives": [candidate.tool_name for candidate in alternatives],
            "parsed_route": (
                {
                    "tool_names": list(parsed_route.tool_names),
                    "reason": parsed_route.reason,
                }
                if parsed_route is not None
                else None
            ),
            "model_response": llm_response,
            "request_id": resolved_request_id,
        }

    def _build_result(
        self,
        *,
        workflow_result: ExecutionResult,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Build public ``ExecutionResult`` from workflow outcomes.

        Args:
            workflow_result: Aggregated workflow runtime result.
            request_id: Stable request identifier for the run.
            dependencies: Dependency payload mapping passed at run-time.

        Returns:
            Final agent execution result with normalized workflow-first envelope fields.

        Raises:
            RuntimeError: If required workflow step outputs are missing.
        """
        select_step = workflow_result.step_results.get("select_route")
        if select_step is None:
            raise RuntimeError("Router single-step workflow missing select_route result.")
        select_output = select_step.output

        raw_model_response = select_output.get("model_response")
        model_response = raw_model_response if isinstance(raw_model_response, LLMResponse) else None
        selected_tool_name_raw = select_output.get("selected_tool_name")
        selected_tool_name = (
            selected_tool_name_raw if isinstance(selected_tool_name_raw, str) else ""
        )
        selected_step_id = self._tool_step_ids.get(selected_tool_name, "")
        selected_step = workflow_result.step_results.get(selected_step_id)

        parsed_route = select_output.get("parsed_route")
        parsed_route_payload = dict(parsed_route) if isinstance(parsed_route, Mapping) else None
        alternatives_raw = select_output.get("alternatives")
        alternatives = (
            list(alternatives_raw)
            if isinstance(alternatives_raw, Sequence)
            and not isinstance(alternatives_raw, (str, bytes))
            else [candidate.tool_name for candidate in self._default_alternatives]
        )

        if not workflow_result.success and not bool(select_output.get("selection_valid", False)):
            if model_response is None:
                raise RuntimeError("Router failure path missing model response payload.")
            fallback = routing_failure_result(
                error=str(
                    select_output.get(
                        "error",
                        "Router model output was invalid. Expected valid route output.",
                    )
                ),
                llm_response=model_response,
                request_id=request_id,
                dependencies=dependencies,
                alternatives=[
                    ToolAlternative(tool_name=name, description="", input_schema={})
                    for name in alternatives
                ],
                parsed_route=_parsed_route_from_mapping(parsed_route_payload),
            )
            output = build_workflow_first_output(
                base_output=fallback.output,
                workflow_result=workflow_result,
                final_output=fallback.output.get("tool_output", {}),
            )
            return ExecutionResult(
                output=output,
                success=False,
                tool_results=[],
                model_response=model_response,
                metadata=dict(fallback.metadata),
            )

        if not selected_tool_name:
            raise RuntimeError("Router single-step workflow missing selected tool name.")
        if selected_step is None:
            raise RuntimeError("Router single-step workflow missing selected tool step output.")

        tool_result = _tool_result_from_step_output(
            step_output=selected_step.output,
            fallback_tool_name=selected_tool_name,
        )
        tool_output = tool_result.result
        base_output: dict[str, object] = {
            "model_text": str(select_output.get("model_text", "")),
            "model_response": (
                dict(select_output.get("model_response_payload"))
                if isinstance(select_output.get("model_response_payload"), Mapping)
                else {}
            ),
            "tool_name": selected_tool_name,
            "tool_names": list(select_output.get("selected_tool_names", []))
            if isinstance(select_output.get("selected_tool_names"), Sequence)
            and not isinstance(select_output.get("selected_tool_names"), (str, bytes))
            else [],
            "tool_input": _mapping_or_empty(select_output.get("tool_input")),
            "tool_output": tool_output,
        }
        output = build_workflow_first_output(
            base_output=base_output,
            workflow_result=workflow_result,
            final_output=tool_output,
        )
        return ExecutionResult(
            output=output,
            success=workflow_result.success and tool_result.ok,
            tool_results=[tool_result],
            model_response=model_response,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "routing": {
                    "source": "model",
                    "alternatives": alternatives,
                    "selected_tool_name": selected_tool_name,
                    "selected_tool_names": base_output["tool_names"],
                    "selected_index": select_output.get("selected_index"),
                    "selected_reason": select_output.get("selected_reason"),
                    "parsed_route": parsed_route_payload,
                },
            },
        )


def _invalid_route_handler(context: Mapping[str, object]) -> Mapping[str, object]:
    """Raise deterministic route validation failure from routed selection output.

    Args:
        context: Workflow step execution context payload.

    Returns:
        Mapping payload for the logic-step contract; this function always raises.

    Raises:
        ValueError: If routed selection output is missing or invalid.
    """
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        raise ValueError("Missing dependency_results for invalid_route step.")
    select_result = dependency_results.get("select_route")
    if not isinstance(select_result, Mapping):
        raise ValueError("Missing select_route result for invalid_route step.")
    select_output = select_result.get("output")
    if not isinstance(select_output, Mapping):
        raise ValueError("Invalid select_route output for invalid_route step.")
    error_text = select_output.get("error")
    resolved_error = (
        str(error_text)
        if isinstance(error_text, str) and error_text.strip()
        else "invalid model route output"
    )
    raise ValueError(resolved_error)


def _build_tool_input_builder(*, expected_tool_name: str) -> ToolStepInputBuilder:
    """Build ``ToolStep.input_builder`` callback for one routed route tool.

    Args:
        expected_tool_name: Tool name bound to the generated builder callback.

    Returns:
        Tool-step input builder that forwards selected route input payloads.
    """

    def _builder(context: Mapping[str, object]) -> Mapping[str, object]:
        """Extract selected route tool input payload from dependency outputs.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Tool input payload for the expected route tool, or an empty mapping.
        """
        dependency_results = context.get("dependency_results")
        if not isinstance(dependency_results, Mapping):
            return {}
        select_result = dependency_results.get("select_route")
        if not isinstance(select_result, Mapping):
            return {}
        select_output = select_result.get("output")
        if not isinstance(select_output, Mapping):
            return {}
        selected_tool_name = select_output.get("selected_tool_name")
        if selected_tool_name != expected_tool_name:
            return {}
        return _mapping_or_empty(select_output.get("tool_input"))

    return _builder


def _mapping_or_empty(value: object) -> dict[str, object]:
    """Normalize optional mapping payloads to plain dictionaries.

    Args:
        value: Candidate mapping value.

    Returns:
        Plain dictionary when ``value`` is a mapping, otherwise an empty dictionary.
    """
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _tool_result_from_step_output(
    *,
    step_output: Mapping[str, object],
    fallback_tool_name: str,
) -> ToolResult:
    """Rebuild typed ``ToolResult`` from ``ToolStep`` serialized output payload.

    Args:
        step_output: Serialized tool step output mapping.
        fallback_tool_name: Tool name used when payload omits explicit tool name.

    Returns:
        Typed ``ToolResult`` reconstructed from serialized step output.
    """
    metadata = step_output.get("metadata")
    artifacts = step_output.get("artifacts")
    warnings = step_output.get("warnings")
    return ToolResult(
        tool_name=str(step_output.get("tool_name", fallback_tool_name)),
        ok=bool(step_output.get("ok", False)),
        result=step_output.get("result", {}),
        artifacts=artifacts if isinstance(artifacts, Sequence) else (),
        warnings=warnings if isinstance(warnings, Sequence) else (),
        error=_tool_error_payload(step_output.get("error")),
        metadata=metadata if isinstance(metadata, Mapping) else {},
    )


def _parsed_route_from_mapping(payload: Mapping[str, object] | None) -> ParsedRoute | None:
    """Convert parsed-route mapping back to helper dataclass when available.

    Args:
        payload: Parsed-route payload mapping from workflow output.

    Returns:
        Parsed route dataclass when payload is valid, otherwise ``None``.
    """
    if payload is None:
        return None
    tool_names = payload.get("tool_names")
    if not isinstance(tool_names, Sequence) or isinstance(tool_names, (str, bytes)):
        return None

    normalized_names = tuple(name for name in tool_names if isinstance(name, str) and name.strip())
    reason = payload.get("reason")
    reason_text = reason if isinstance(reason, str) else None
    return ParsedRoute(tool_names=normalized_names, reason=reason_text)


def _tool_error_payload(value: object) -> ToolError | Mapping[str, object] | str | None:
    """Narrow raw tool error payload into ``ToolResult`` accepted input types.

    Args:
        value: Candidate tool error payload.

    Returns:
        Value narrowed to accepted ``ToolResult.error`` input types.
    """
    if isinstance(value, (ToolError, str)):
        return value
    if isinstance(value, Mapping):
        return value
    return None


def _tool_step_id(tool_name: str) -> str:
    """Normalize tool names into stable workflow step ids.

    Args:
        tool_name: Tool name to normalize.

    Returns:
        Stable workflow step identifier for routed invocation.
    """
    sanitized = [ch if ch.isalnum() else "_" for ch in tool_name]
    suffix = "".join(sanitized).strip("_") or "tool"
    return f"invoke_{suffix}"


__all__ = [
    "SingleStepToolRouterAgent",
]
