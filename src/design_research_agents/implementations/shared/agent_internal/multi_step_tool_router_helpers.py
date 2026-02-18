"""Shared helpers for the multi-step tool router implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.contracts.agent import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.tools import ToolResult
from design_research_agents.implementations.shared.agent_internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.implementations.shared.agent_internal.result_builders import (
    build_failure_result,
)
from design_research_agents.implementations.shared.agent_internal.router_agent_helpers import (
    ToolAlternative,
)


@dataclass(slots=True, frozen=True)
class ToolRouterStepDecision:
    """Parsed controller output for one tool-router step."""

    action: str
    """Normalized controller action (``TOOL_CALL`` or ``STOP``)."""
    tool_names: tuple[str, ...]
    """Ordered deduplicated tool names for ``TOOL_CALL`` actions."""
    tool_input: dict[str, object] | None
    """Optional tool input payload for ``TOOL_CALL`` actions."""
    final_output: dict[str, object] | None
    """Optional normalized final output payload for ``STOP`` actions."""
    reason: str
    """Controller-provided decision rationale."""
    source: str
    """Decision source label used for tracing and termination mapping."""


def parse_tool_router_step_decision(raw_text: str) -> ToolRouterStepDecision | None:
    """Parse one JSON controller response into a normalized step decision.

    Args:
        raw_text: Raw controller output text produced by the model.

    Returns:
        Normalized step decision, or ``None`` when payload validation fails.
    """
    parsed = _parse_json_mapping(raw_text)
    if parsed is None:
        return None

    raw_action = parsed.get("action")
    action = raw_action.strip().upper() if isinstance(raw_action, str) else ""
    if action not in {"TOOL_CALL", "STOP"}:
        return None

    tool_names = parse_tool_names(parsed)
    if action == "TOOL_CALL" and not tool_names:
        return None

    raw_tool_input = parsed.get("tool_input")
    tool_input = dict(raw_tool_input) if isinstance(raw_tool_input, Mapping) else None
    raw_final_output = parsed.get("final_output")
    final_output = normalize_output_dict(raw_final_output) if raw_final_output is not None else None
    reason = str(parsed.get("reason", "model decision"))

    return ToolRouterStepDecision(
        action=action,
        tool_names=tool_names,
        tool_input=tool_input,
        final_output=final_output,
        reason=reason,
        source="model",
    )


def parse_tool_names(parsed: Mapping[str, object]) -> tuple[str, ...]:
    """Parse one or many selected tool names from a model response mapping.

    Args:
        parsed: Model response payload mapping.

    Returns:
        Ordered deduplicated tool names, or an empty tuple when unavailable.
    """
    raw_tool_names = parsed.get("tool_names")
    if isinstance(raw_tool_names, Sequence) and not isinstance(raw_tool_names, (str, bytes)):
        normalized_names = [
            raw_name.strip()
            for raw_name in raw_tool_names
            if isinstance(raw_name, str) and raw_name.strip()
        ]
        if normalized_names:
            return tuple(dict.fromkeys(normalized_names))
    return ()


def resolve_selected_tool(
    *,
    alternatives: Sequence[ToolAlternative],
    tool_names: Sequence[str],
) -> tuple[str, int] | None:
    """Resolve the first selected tool that exists in runtime alternatives.

    Args:
        alternatives: Available tool alternatives from runtime configuration.
        tool_names: Candidate names emitted by the controller.

    Returns:
        Tuple of normalized tool name and its index in ``alternatives``, or ``None``.
    """
    for tool_name in tool_names:
        for index, alternative in enumerate(alternatives):
            if alternative.tool_name != tool_name:
                continue
            return alternative.tool_name, index
    return None


def normalize_output_dict(raw_output: object) -> dict[str, object]:
    """Normalize one output payload into a mapping for stable agent output shape.

    Args:
        raw_output: Raw model-provided output payload.

    Returns:
        Mapping payload used in deterministic agent output.
    """
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
) -> ExecutionResult:
    """Build a standardized failure payload for router step failures.

    Args:
        error: Human-readable error description.
        model_response: Optional model response associated with the failure.
        tool_results: Tool results accumulated before failure.
        request_id: Request id for traceability metadata.
        dependencies: Dependency mapping attached to output metadata.
        metadata: Additional metadata to merge into the failure payload.
        output: Structured output payload for downstream consumers.

    Returns:
        Execution result with ``success=False`` and normalized failure metadata.
    """
    return build_failure_result(
        error=error,
        model_response=model_response,
        tool_results=tool_results,
        request_id=request_id,
        dependencies=dependencies,
        metadata=metadata,
        output=output,
    )
