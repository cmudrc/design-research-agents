"""Shared helpers for the multi-step tool router implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.agent.internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.agent.internal.result_builders import build_failure_result
from design_research_agents.agent.internal.router_agent_helpers import ToolAlternative
from design_research_agents.contracts.agent import AgentResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.tools import ToolResult


@dataclass(slots=True, frozen=True)
class ToolRouterStepDecision:
    """Parsed controller output for one tool-router step."""

    action: str
    tool_names: tuple[str, ...]
    tool_input: dict[str, object] | None
    final_output: dict[str, object] | None
    reason: str
    source: str


def parse_tool_router_step_decision(raw_text: str) -> ToolRouterStepDecision | None:
    """Parse one JSON controller response into a normalized step decision."""
    parsed = _parse_json_mapping(raw_text)
    if parsed is None:
        return None

    raw_action = parsed.get("action", parsed.get("decision"))
    action = raw_action.strip().upper() if isinstance(raw_action, str) else ""
    if not action:
        if bool(parsed.get("stop")):
            action = "STOP"
        elif parsed.get("tool_names") is not None or parsed.get("tool_name") is not None:
            action = "TOOL_CALL"
    if action not in {"TOOL_CALL", "STOP"}:
        return None

    tool_names = parse_tool_names(parsed)
    if action == "TOOL_CALL" and not tool_names:
        return None

    raw_tool_input = parsed.get("tool_input")
    tool_input = dict(raw_tool_input) if isinstance(raw_tool_input, Mapping) else None
    raw_final_output = parsed.get("final_output")
    final_output = normalize_output_dict(raw_final_output) if raw_final_output is not None else None
    reason = str(parsed.get("reason", parsed.get("thought", "model decision")))

    return ToolRouterStepDecision(
        action=action,
        tool_names=tool_names,
        tool_input=tool_input,
        final_output=final_output,
        reason=reason,
        source="model",
    )


def parse_tool_names(parsed: Mapping[str, object]) -> tuple[str, ...]:
    """Parse one or many selected tool names from a model response mapping."""
    raw_tool_names = parsed.get("tool_names")
    if isinstance(raw_tool_names, Sequence) and not isinstance(raw_tool_names, (str, bytes)):
        normalized_names = [
            raw_name.strip()
            for raw_name in raw_tool_names
            if isinstance(raw_name, str) and raw_name.strip()
        ]
        if normalized_names:
            return tuple(dict.fromkeys(normalized_names))

    raw_tool_name = parsed.get("tool_name", parsed.get("name", parsed.get("selection")))
    if isinstance(raw_tool_name, str) and raw_tool_name.strip():
        return (raw_tool_name.strip(),)
    return ()


def resolve_selected_tool(
    *,
    alternatives: Sequence[ToolAlternative],
    tool_names: Sequence[str],
) -> tuple[str, int] | None:
    """Resolve the first selected tool that exists in runtime alternatives."""
    for tool_name in tool_names:
        for index, alternative in enumerate(alternatives):
            if alternative.tool_name != tool_name:
                continue
            return alternative.tool_name, index
    return None


def normalize_output_dict(raw_output: object) -> dict[str, object]:
    """Normalize one output payload into a mapping for stable agent output shape."""
    if isinstance(raw_output, Mapping):
        return dict(raw_output)
    if raw_output is None:
        return {}
    return {"value": raw_output}


def failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: Sequence[ToolResult],
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
) -> AgentResult:
    """Build a standardized failure payload for router step failures."""
    return build_failure_result(
        error=error,
        model_response=model_response,
        tool_results=tool_results,
        request_id=request_id,
        dependencies=dependencies,
        metadata=metadata,
        output=output,
    )
