"""Helper utilities for JSON action-step tool-calling agents."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from design_research_agents.contracts.llm import LLMClient, LLMRequest, LLMResponse
from design_research_agents.contracts.tools import ToolSpec
from design_research_agents.implementations.shared.agent_internal.input_parsing import (
    load_json_mapping,
    parse_json_mapping,
)
from design_research_agents.implementations.shared.agent_internal.prompt_overrides import (
    render_template_text,
)
from design_research_agents.implementations.shared.agent_internal.response_schemas import (
    build_tool_call_response_schema,
)
from design_research_agents.implementations.shared.agent_internal.tool_input import (
    resolve_known_tool_input,
)


@dataclass(slots=True, frozen=True)
class ToolChoice:
    """Normalized tool option used by planning and validation logic."""

    tool_name: str
    """Field value for ``tool_name``."""
    description: str
    """Field value for ``description``."""
    input_schema: dict[str, object]
    """Field value for ``input_schema``."""


def extract_tool_choices(
    *,
    tool_specs: Mapping[str, ToolSpec],
    allowed_tool_names: Sequence[str] | None = None,
) -> list[ToolChoice]:
    """Extract normalized tool choices from runtime specs.

    Args:
        tool_specs: Input value for this parameter.
        allowed_tool_names: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when validation or execution fails.
    """
    allowed_name_set = set(allowed_tool_names or [])
    filtered_specs = [
        spec
        for spec in tool_specs.values()
        if not allowed_name_set or spec.name in allowed_name_set
    ]
    if filtered_specs:
        return [
            ToolChoice(
                tool_name=spec.name,
                description=spec.description,
                input_schema=dict(spec.input_schema),
            )
            for spec in filtered_specs
        ]

    raise ValueError(
        "JSON action step runner requires at least one tool in ToolRuntime.list_tools()."
    )


def build_tool_call_prompt(*, prompt: str, choices_block: str, prompt_template: str) -> str:
    """Build prompt asking model to select tool and structured arguments.

    Args:
        prompt: Input value for this parameter.
        choices_block: Input value for this parameter.
        prompt_template: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return render_template_text(
        template_text=prompt_template,
        variables={
            "choices_block": choices_block,
            "user_prompt": prompt,
        },
        field_name="user_prompt_template",
    )


def resolve_allowed_tool_names(
    *,
    runtime_specs: Mapping[str, ToolSpec],
    allowed_tools: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Resolve tool allowlist against runtime specs.

    Args:
        runtime_specs: Input value for this parameter.
        allowed_tools: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when validation or execution fails.
    """
    if allowed_tools is None:
        return None

    resolved_names = [
        tool_name.strip()
        for tool_name in allowed_tools
        if isinstance(tool_name, str) and tool_name.strip() in runtime_specs
    ]
    deduped_names = tuple(dict.fromkeys(resolved_names))
    if not deduped_names:
        raise ValueError("allowed_tools did not match any runtime tools.")
    return deduped_names


def build_tool_choices_text(*, choices: Sequence[ToolChoice]) -> str:
    """Build formatted runtime tool choices text.

    Args:
        choices: Input value for this parameter.

    Returns:
        Computed return value.
    """
    choice_lines: list[str] = []
    for choice in choices:
        choice_lines.append(
            "\n".join(
                [
                    f"- tool_name: {choice.tool_name}",
                    f"  description: {choice.description or '(none)'}",
                    f"  input_schema: {json.dumps(choice.input_schema, sort_keys=True)}",
                ]
            )
        )
    return "\n".join(choice_lines)


def clone_tool_choice(choice: ToolChoice) -> ToolChoice:
    """Clone one tool choice so run-local payloads remain isolated.

    Args:
        choice: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return ToolChoice(
        tool_name=choice.tool_name,
        description=choice.description,
        input_schema=dict(choice.input_schema),
    )


def request_tool_call_response(
    *,
    llm_client: LLMClient,
    llm_request: LLMRequest,
) -> LLMResponse:
    """Dispatch one tool-call planning request to the LLM client.

    Args:
        llm_client: Input value for this parameter.
        llm_request: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return cast(Callable[[LLMRequest], LLMResponse], llm_client.generate)(llm_request)


def tool_call_response_schema(available_tool_names: Sequence[str]) -> dict[str, object]:
    """Build the strict tool-call response schema for available tools.

    Args:
        available_tool_names: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return build_tool_call_response_schema(
        tool_names=available_tool_names,
    )


def parse_tool_call_from_response(
    llm_response: LLMResponse,
) -> dict[str, object] | None:
    """Extract first structured tool call payload from provider tool-call metadata.

    Args:
        llm_response: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if not llm_response.tool_calls:
        return None
    call = llm_response.tool_calls[0]
    try:
        tool_input = json.loads(call.arguments_json)
    except json.JSONDecodeError:
        tool_input = call.arguments_json
    return {
        "tool_name": call.name,
        "tool_input": tool_input,
        "call_id": call.call_id,
    }


def parse_tool_call(raw_text: str) -> dict[str, object] | None:
    """Parse tool-call JSON payload from model text output.

    Args:
        raw_text: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return parse_json_mapping(raw_text)


def select_tool_choice(
    *,
    parsed_tool_call: Mapping[str, object] | None,
    choices: Sequence[ToolChoice],
) -> tuple[ToolChoice, str, str] | None:
    """Select a validated tool choice from structured model output.

    Args:
        parsed_tool_call: Input value for this parameter.
        choices: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if parsed_tool_call is None:
        return None

    selection_payload = _resolve_selection_payload(parsed_tool_call)
    if selection_payload is None:
        return None
    selected_name, source, reason = selection_payload
    allowed_names = {choice.tool_name for choice in choices}
    if selected_name not in allowed_names:
        return None
    selected_choice = next(choice for choice in choices if choice.tool_name == selected_name)
    return selected_choice, source, reason


def _resolve_selection_payload(
    parsed_tool_call: Mapping[str, object],
) -> tuple[str, str, str] | None:
    """Resolve normalized tool-name selection payload from structured model output.

    Args:
        parsed_tool_call: Parsed tool-call mapping from model output text or metadata.

    Returns:
        Tuple of ``(tool_name, source, reason)`` when a selection can be resolved.
    """
    raw_action = parsed_tool_call.get("action")
    if isinstance(raw_action, str):
        normalized_action = raw_action.strip().upper()
        if normalized_action and normalized_action != "TOOL_CALL":
            return None

    raw_tool_name = parsed_tool_call.get("tool_name")
    if isinstance(raw_tool_name, str) and raw_tool_name.strip():
        return raw_tool_name.strip(), "model", "validated model tool_name"

    raw_legacy_name = parsed_tool_call.get("name")
    if isinstance(raw_legacy_name, str) and raw_legacy_name.strip():
        return (
            raw_legacy_name.strip(),
            "model_legacy_name",
            "validated legacy `name` selection",
        )

    raw_tool_names = parsed_tool_call.get("tool_names")
    if isinstance(raw_tool_names, Sequence) and not isinstance(raw_tool_names, (str, bytes)):
        for candidate_name in raw_tool_names:
            if isinstance(candidate_name, str) and candidate_name.strip():
                return (
                    candidate_name.strip(),
                    "model_legacy_tool_names",
                    "validated legacy `tool_names` selection",
                )
    return None


def resolve_tool_input(
    *,
    selected_choice: ToolChoice,
    parsed_tool_call: Mapping[str, object] | None,
    input_payload: Mapping[str, object],
) -> dict[str, object]:
    """Resolve final tool input from model payload, run input, or heuristics.

    Args:
        selected_choice: Input value for this parameter.
        parsed_tool_call: Input value for this parameter.
        input_payload: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if parsed_tool_call is not None:
        raw_tool_input = parsed_tool_call.get(
            "tool_input",
            parsed_tool_call.get("arguments", parsed_tool_call.get("args")),
        )
        normalized_from_model = coerce_tool_input(raw_tool_input)
        if normalized_from_model:
            return normalized_from_model

    raw_tool_input = input_payload.get("tool_input")
    normalized_from_input = coerce_tool_input(raw_tool_input)
    if normalized_from_input:
        return normalized_from_input

    known_tool_input = resolve_known_tool_input(
        tool_name=selected_choice.tool_name,
        input_payload=input_payload,
    )
    if known_tool_input is not None:
        return known_tool_input

    return {}


def coerce_tool_input(raw_tool_input: object) -> dict[str, object] | None:
    """Convert raw tool-input payload into a JSON-like dictionary when possible.

    Args:
        raw_tool_input: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if isinstance(raw_tool_input, Mapping):
        return dict(raw_tool_input)
    if isinstance(raw_tool_input, str):
        parsed = load_json_mapping(raw_tool_input)
        if parsed is not None:
            return parsed
    return None


__all__ = [
    "ToolChoice",
    "build_tool_call_prompt",
    "build_tool_choices_text",
    "clone_tool_choice",
    "extract_tool_choices",
    "parse_tool_call",
    "parse_tool_call_from_response",
    "request_tool_call_response",
    "resolve_allowed_tool_names",
    "resolve_tool_input",
    "select_tool_choice",
    "tool_call_response_schema",
]
