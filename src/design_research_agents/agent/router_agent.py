"""Router agent implementation that selects one tool alternative per request.

The router asks the model to choose a route from runtime-backed alternatives,
requires a structured route payload, executes the selected tool, and returns
both model and tool artifacts in a single ``AgentResult``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.agent._model_resolution import resolve_agent_model
from design_research_agents.agent._prompt_alternatives import (
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    resolve_alternatives_prompt_target,
)
from design_research_agents.agent._response_schemas import (
    build_router_selection_response_schema,
    clone_response_schema,
)
from design_research_agents.agent._run_options import (
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
from design_research_agents.contracts.tools import ToolRuntime, ToolSpec
from design_research_agents.prompts import load_prompt, render_prompt
from design_research_agents.tracing import (
    emit_guardrail_decision,
    emit_router_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


@dataclass(slots=True, frozen=True)
class _ToolAlternative:
    """Normalized candidate route used by routing prompt and validation logic.

    Attributes:
        tool_name: Runtime tool identifier that can be invoked.
        description: Human-readable routing description for model planning.
        input_schema: JSON-schema-like shape expected by the tool.
    """

    tool_name: str
    description: str
    input_schema: dict[str, object]


@dataclass(slots=True, frozen=True)
class _ParsedRoute:
    """Parsed model payload describing a discrete route selection.

    Attributes:
        selection: Selected route identifier (index or tool name).
        reason: Optional model-provided rationale for the route decision.
    """

    selection: int | str
    reason: str | None


class RouterAgent(Agent):
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
        model: str | None = None,
        default_tool_name: str = "text_stats_tool",
    ) -> None:
        """Initialize a router agent with injected runtime dependencies.

        Args:
            llm_client: LLM client used for prompt execution.
            tool_runtime: Tool runtime used for tool invocation.
            model: Optional model override applied to all runs when provided.
            default_tool_name: Fallback tool used when no alternatives are supplied.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._model = model
        self._default_tool_name = default_tool_name
        self._runtime_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        self._compiled_runtime_alternatives = _compile_runtime_alternatives(
            tool_specs=self._runtime_specs
        )
        self._default_alternatives = _extract_alternatives(
            runtime_specs=self._runtime_specs,
            compiled_runtime_alternatives=self._compiled_runtime_alternatives,
            default_tool_name=self._default_tool_name,
        )
        self._default_route_response_schema = _route_response_schema(
            alternatives=self._default_alternatives,
        )

    def run(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run one model route-selection call and one routed tool invocation.

        Invalid model routing output is treated as a hard failure result instead
        of triggering deterministic routing fallbacks.

        Args:
            input: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(input)
        trace_scope = start_trace_run(
            agent_name="RouterAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
        )
        prompt = _extract_prompt(normalized_input)
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
            input_payload=normalized_input,
            init_model=self._model,
        )
        alternatives = [
            _clone_alternative(alternative) for alternative in self._default_alternatives
        ]
        alternatives_prompt_target = resolve_alternatives_prompt_target(
            input_payload=normalized_input
        )
        routes_text = _build_routes_text(alternatives=alternatives)
        routes_block = build_user_prompt_alternatives_block(
            section_label="Available routes",
            alternatives_text=routes_text,
            target=alternatives_prompt_target,
        )
        user_prompt = _build_route_prompt(prompt=prompt, routes_block=routes_block)
        system_prompt = load_prompt("router_system")
        if alternatives_prompt_target == "system":
            system_prompt = append_alternatives_block(
                prompt_text=system_prompt,
                section_label="Available routes",
                alternatives_text=routes_text,
            )

        llm_params = LLMChatParams(
            response_schema=clone_response_schema(self._default_route_response_schema),
            provider_options={"agent": "RouterAgent", "phase": "route_select"},
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
        model_span_id = start_model_call(
            model=resolved_model,
            messages=messages,
            params=llm_params,
            metadata={"agent": "RouterAgent", "phase": "route_select"},
        )
        try:
            llm_response = self._llm_client.chat(messages, model=resolved_model, params=llm_params)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            finish_trace_run(trace_scope, error=str(exc))
            raise
        finish_model_call(model_span_id, response=llm_response)
        parsed_route = _parse_route_response(llm_response.text)

        route_resolution = _resolve_model_route(
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
            result = _routing_failure_result(
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
        tool_input = _resolve_tool_input(
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
            "tool_output": tool_result.output,
        }
        result = AgentResult(
            output=output,
            success=tool_result.success,
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
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Emit a deterministic stream wrapper around ``run``.

        The wrapper currently emits one full-text delta event followed by a
        completion event that carries the full ``AgentResult`` payload.

        Args:
            input: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Yields:
            Streaming events through completion.
        """
        result = self.run(input, request_id=request_id, dependencies=dependencies)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)


def _routing_failure_result(
    *,
    error: str,
    llm_response: LLMResponse,
    request_id: str,
    dependencies: Mapping[str, object],
    alternatives: Sequence[_ToolAlternative],
    parsed_route: _ParsedRoute | None,
) -> AgentResult:
    """Build a structured failure result for invalid/incomplete model routing.

    The returned payload preserves routing metadata and parsed model artifacts
    so callers can inspect why route validation failed.

    Args:
        error: Failure message describing the routing issue.
        llm_response: Model response payload.
        request_id: Request identifier for tracing.
        dependencies: Dependency payload mapping.
        alternatives: Available tool alternatives.
        parsed_route: Parsed route payload, if available.

    Returns:
        Agent result payload describing the routing failure.
    """
    output: dict[str, object] = {
        "error": error,
        "model_text": llm_response.text,
        "model_response": {},
        "tool_name": None,
        "selected_alternative_index": None,
        "tool_input": {},
        "tool_output": {},
    }
    return AgentResult(
        output=output,
        success=False,
        tool_results=[],
        model_response=llm_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            "stage": "routing",
            "routing": {
                "source": "model_invalid",
                "alternatives": [candidate.tool_name for candidate in alternatives],
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


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from run input.

    Falls back to ``text`` and then a default string when missing.

    Args:
        input_payload: Normalized run input payload mapping.

    Returns:
        Prompt text for the run.
    """
    raw_prompt = input_payload.get(
        "prompt", input_payload.get("text", "Provide a concise response.")
    )
    return str(raw_prompt)


def _extract_alternatives(
    *,
    runtime_specs: Mapping[str, ToolSpec],
    compiled_runtime_alternatives: Sequence[_ToolAlternative],
    default_tool_name: str,
) -> list[_ToolAlternative]:
    """Return routing alternatives compiled from runtime tool specifications.

    Args:
        runtime_specs: Tool specs available in the runtime.
        compiled_runtime_alternatives: Cached runtime alternatives.
        default_tool_name: Fallback tool name when no alternatives exist.

    Returns:
        List of tool alternatives for routing.
    """
    if compiled_runtime_alternatives:
        return [_clone_alternative(alternative) for alternative in compiled_runtime_alternatives]

    default_runtime_spec = runtime_specs.get(default_tool_name)
    return [
        _ToolAlternative(
            tool_name=default_tool_name,
            description=(
                default_runtime_spec.description
                if default_runtime_spec is not None
                else "Default fallback route."
            ),
            input_schema=(
                dict(default_runtime_spec.input_schema)
                if default_runtime_spec is not None
                else {"type": "object"}
            ),
        )
    ]


def _clone_alternative(alternative: _ToolAlternative) -> _ToolAlternative:
    """Clone one alternative to keep run-level payload mutations isolated.

    Args:
        alternative: Alternative to clone.

    Returns:
        Cloned alternative instance.
    """
    return _ToolAlternative(
        tool_name=alternative.tool_name,
        description=alternative.description,
        input_schema=dict(alternative.input_schema),
    )


def _compile_runtime_alternatives(
    *,
    tool_specs: Mapping[str, ToolSpec],
) -> tuple[_ToolAlternative, ...]:
    """Compile default routing alternatives directly from runtime tool specs.

    Compiled alternatives are cached at initialization and cloned per run.

    Args:
        tool_specs: Tool specs available in the runtime.

    Returns:
        Tuple of compiled tool alternatives.
    """
    return tuple(
        _ToolAlternative(
            tool_name=spec.name,
            description=spec.description,
            input_schema=dict(spec.input_schema),
        )
        for spec in tool_specs.values()
    )


def _build_route_prompt(
    *,
    prompt: str,
    routes_block: str,
) -> str:
    """Build the route-selection user prompt consumed by the model.

    Args:
        prompt: User prompt text.
        routes_block: Pre-rendered routes block text.

    Returns:
        Rendered route-selection prompt text.
    """
    return render_prompt(
        "router_user_route",
        variables={
            "routes_block": routes_block,
            "user_prompt": prompt,
        },
    )


def _build_routes_text(
    *,
    alternatives: Sequence[_ToolAlternative],
) -> str:
    """Build formatted runtime route alternatives text.

    Args:
        alternatives: Tool alternatives to render.

    Returns:
        Formatted routes block text.
    """
    route_lines: list[str] = []
    for index, alternative in enumerate(alternatives):
        route_lines.append(
            "\n".join(
                [
                    f"- selection_index: {index}",
                    f"  tool_name: {alternative.tool_name}",
                    f"  description: {alternative.description or '(none)'}",
                    f"  input_schema: {json.dumps(alternative.input_schema, sort_keys=True)}",
                ]
            )
        )
    return "\n".join(route_lines)


def _route_response_schema(
    *,
    alternatives: Sequence[_ToolAlternative],
) -> dict[str, object]:
    """Build route-selection schema from runtime-derived alternatives.

    Args:
        alternatives: Tool alternatives used to constrain selection.

    Returns:
        JSON-schema-like mapping for route selection.
    """
    return build_router_selection_response_schema(
        alternative_identifiers=[alternative.tool_name for alternative in alternatives]
    )


def _parse_route_response(raw_text: str) -> _ParsedRoute | None:
    """Parse model route JSON payload from raw text output.

    The parser accepts either strict-JSON responses or JSON objects embedded in
    surrounding text.

    Args:
        raw_text: Raw model response text.

    Returns:
        Parsed route payload or ``None`` when parsing fails.
    """
    parsed = _load_json_mapping(raw_text)
    if parsed is None:
        # Allow extra surrounding text by scanning for the first JSON object.
        decoder = json.JSONDecoder()
        for index, character in enumerate(raw_text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                parsed = dict(value)
                break

    if parsed is None:
        return None

    raw_selection = parsed.get(
        "selection",
        parsed.get(
            "selected_alternative_index",
            parsed.get("tool_name", parsed.get("name")),
        ),
    )
    if not isinstance(raw_selection, (int, str)):
        return None
    if isinstance(raw_selection, int):
        selection: int | str = raw_selection
    else:
        normalized_selection = raw_selection.strip()
        if not normalized_selection:
            return None
        selection = normalized_selection

    return _ParsedRoute(
        selection=selection,
        reason=(
            str(parsed["reason"]) if "reason" in parsed and parsed["reason"] is not None else None
        ),
    )


def _load_json_mapping(raw_text: str) -> dict[str, object] | None:
    """Load text as a JSON mapping.

    Returns ``None`` when the text is invalid JSON or not an object.

    Args:
        raw_text: Raw text to parse as JSON.

    Returns:
        Parsed JSON mapping or ``None`` when invalid.
    """
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _resolve_model_route(
    *,
    parsed_route: _ParsedRoute | None,
    alternatives: Sequence[_ToolAlternative],
) -> tuple[_ToolAlternative, int, str] | None:
    """Resolve and validate model-selected route against available alternatives.

    Supports both integer index selections and string tool-name identifiers.

    Args:
        parsed_route: Parsed route payload, if available.
        alternatives: Available tool alternatives.

    Returns:
        Tuple of selected alternative, index, and reason, or ``None`` when invalid.
    """
    if parsed_route is None:
        return None

    if isinstance(parsed_route.selection, int):
        selected_index = parsed_route.selection
        if 0 <= selected_index < len(alternatives):
            selected_alternative = alternatives[selected_index]
            return (
                selected_alternative,
                selected_index,
                parsed_route.reason or "validated model selection index",
            )
        return None

    selected_identifier = parsed_route.selection
    for index, alternative in enumerate(alternatives):
        if alternative.tool_name != selected_identifier:
            continue
        return (
            alternative,
            index,
            (parsed_route.reason or "validated model selection identifier"),
        )

    return None


def _resolve_tool_input(
    *,
    tool_name: str,
    input_payload: Mapping[str, object],
) -> dict[str, object]:
    """Resolve tool input from run payload and tool-specific heuristics.

    Args:
        tool_name: Tool name for which input is being resolved.
        input_payload: Normalized run input payload mapping.

    Returns:
        Tool input mapping.
    """
    raw_tool_input = input_payload.get("tool_input")
    if isinstance(raw_tool_input, Mapping):
        return dict(raw_tool_input)

    if tool_name == "calculator_tool":
        prompt_text = str(input_payload.get("prompt", input_payload.get("text", "")))
        expression = _infer_expression(
            input_payload=input_payload,
            prompt=prompt_text,
        )
        return {"expression": expression}

    if tool_name == "text_stats_tool":
        analysis_text = input_payload.get("analysis_text")
        if analysis_text is not None:
            return {"text": str(analysis_text)}
        return {"text": _extract_prompt(input_payload)}

    return {}


def _infer_expression(*, input_payload: Mapping[str, object], prompt: str) -> str:
    """Infer calculator expression from payload fields and prompt text.

    Explicit expressions win; otherwise a regex candidate is extracted from the
    prompt before falling back to the full prompt text.

    Args:
        input_payload: Normalized run input payload mapping.
        prompt: User prompt text.

    Returns:
        Inferred arithmetic expression string.
    """
    explicit_expression = input_payload.get("expression")
    if explicit_expression is not None:
        return str(explicit_expression)

    text_expression = input_payload.get("text")
    if text_expression is not None:
        text_value = str(text_expression)
        if any(operator in text_value for operator in "+-*/%"):
            return text_value

    match = re.search(r"(\(?-?\d[\d\s\.\+\-\*\/%\(\)]*\d\)?)", prompt)
    if match is not None:
        expression = match.group(1).strip()
        if expression and any(operator in expression for operator in "+-*/%"):
            return expression

    return prompt
