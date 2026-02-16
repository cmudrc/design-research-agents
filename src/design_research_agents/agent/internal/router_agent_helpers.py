"""Helper utilities for single-step router agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.agent.internal.input_parsing import parse_json_mapping
from design_research_agents.agent.internal.prompt_overrides import render_template_text
from design_research_agents.agent.internal.response_schemas import (
    build_router_selection_response_schema,
)
from design_research_agents.agent.internal.tool_input import (
    extract_prompt,
    resolve_known_tool_input,
)
from design_research_agents.contracts.agent import AgentResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.tools import ToolSpec


@dataclass(slots=True, frozen=True)
class ToolAlternative:
    """Normalized candidate route used by routing prompt and validation logic."""

    tool_name: str
    description: str
    input_schema: dict[str, object]


@dataclass(slots=True, frozen=True)
class ParsedRoute:
    """Parsed model payload describing a discrete route selection."""

    selection: int | str
    reason: str | None


def routing_failure_result(
    *,
    error: str,
    llm_response: LLMResponse,
    request_id: str,
    dependencies: Mapping[str, object],
    alternatives: Sequence[ToolAlternative],
    parsed_route: ParsedRoute | None,
) -> AgentResult:
    """Build a structured failure result for invalid/incomplete model routing."""
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


def extract_alternatives(
    *,
    runtime_specs: Mapping[str, ToolSpec],
    compiled_runtime_alternatives: Sequence[ToolAlternative],
) -> list[ToolAlternative]:
    """Return routing alternatives compiled from runtime tool specifications."""
    del runtime_specs
    if compiled_runtime_alternatives:
        return [clone_alternative(alternative) for alternative in compiled_runtime_alternatives]

    raise ValueError(
        "SingleStepRouterAgent requires at least one tool in ToolRuntime.list_tools()."
    )


def clone_alternative(alternative: ToolAlternative) -> ToolAlternative:
    """Clone one alternative to keep run-level payload mutations isolated."""
    return ToolAlternative(
        tool_name=alternative.tool_name,
        description=alternative.description,
        input_schema=dict(alternative.input_schema),
    )


def compile_runtime_alternatives(
    *,
    tool_specs: Mapping[str, ToolSpec],
    allowed_route_names: Sequence[str] | None = None,
) -> tuple[ToolAlternative, ...]:
    """Compile default routing alternatives directly from runtime tool specs."""
    allowed_name_set = set(allowed_route_names or [])
    return tuple(
        ToolAlternative(
            tool_name=spec.name,
            description=spec.description,
            input_schema=dict(spec.input_schema),
        )
        for spec in tool_specs.values()
        if not allowed_name_set or spec.name in allowed_name_set
    )


def resolve_allowed_route_names(
    *,
    runtime_specs: Mapping[str, ToolSpec],
    allowed_routes: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Resolve route allowlist against runtime specs."""
    if allowed_routes is None:
        return None

    resolved_names = [
        route_name.strip()
        for route_name in allowed_routes
        if isinstance(route_name, str) and route_name.strip() in runtime_specs
    ]
    deduped_names = tuple(dict.fromkeys(resolved_names))
    if not deduped_names:
        raise ValueError("allowed_routes did not match any runtime routes.")
    return deduped_names


def build_route_prompt(
    *,
    prompt: str,
    routes_block: str,
    prompt_template: str,
) -> str:
    """Build the route-selection user prompt consumed by the model."""
    return render_template_text(
        template_text=prompt_template,
        variables={
            "routes_block": routes_block,
            "user_prompt": prompt,
        },
        field_name="user_prompt_template",
    )


def build_routes_text(
    *,
    alternatives: Sequence[ToolAlternative],
) -> str:
    """Build formatted runtime route alternatives text."""
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


def route_response_schema(
    *,
    alternatives: Sequence[ToolAlternative],
) -> dict[str, object]:
    """Build route-selection schema from runtime-derived alternatives."""
    return build_router_selection_response_schema(
        alternative_identifiers=[alternative.tool_name for alternative in alternatives]
    )


def parse_route_response(raw_text: str) -> ParsedRoute | None:
    """Parse model route JSON payload from raw text output."""
    parsed = parse_json_mapping(raw_text)
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

    return ParsedRoute(
        selection=selection,
        reason=(
            str(parsed["reason"]) if "reason" in parsed and parsed["reason"] is not None else None
        ),
    )


def resolve_model_route(
    *,
    parsed_route: ParsedRoute | None,
    alternatives: Sequence[ToolAlternative],
) -> tuple[ToolAlternative, int, str] | None:
    """Resolve and validate model-selected route against available alternatives."""
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


def resolve_tool_input(
    *,
    tool_name: str,
    input_payload: Mapping[str, object],
) -> dict[str, object]:
    """Resolve tool input from run payload and tool-specific heuristics."""
    raw_tool_input = input_payload.get("tool_input")
    if isinstance(raw_tool_input, Mapping):
        return dict(raw_tool_input)

    known_tool_input = resolve_known_tool_input(
        tool_name=tool_name,
        input_payload=input_payload,
        text_fallback=extract_prompt(input_payload),
    )
    if known_tool_input is not None:
        return known_tool_input

    prompt_text = extract_prompt(input_payload)
    return {
        "prompt": prompt_text,
        "request": prompt_text,
    }


__all__ = [
    "ParsedRoute",
    "ToolAlternative",
    "build_route_prompt",
    "build_routes_text",
    "clone_alternative",
    "compile_runtime_alternatives",
    "extract_alternatives",
    "parse_route_response",
    "resolve_allowed_route_names",
    "resolve_model_route",
    "resolve_tool_input",
    "route_response_schema",
    "routing_failure_result",
]
