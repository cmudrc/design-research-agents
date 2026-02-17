"""Shared helpers for backend prompt formatting and parsing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from design_research_agents.contracts.llm import LLMMessage, ToolCall, Usage


def messages_to_prompt(messages: Sequence[LLMMessage]) -> str:
    """Combine messages into a plain prompt for non-chat backends.

    Args:
        messages: Parameter value.

    Returns:
        The resulting value.
    """
    segments: list[str] = []
    for message in messages:
        role = message.role
        content = message.content
        segments.append(f"{role}: {content}")
    return "\n".join(segments)


def parse_usage(payload: Any) -> Usage | None:
    """Parse OpenAI-style usage fields into a typed usage payload.

    Args:
        payload: Parameter value.

    Returns:
        The resulting value.
    """
    if not isinstance(payload, dict):
        return None
    prompt_tokens = _coerce_int(payload.get("prompt_tokens"))
    completion_tokens = _coerce_int(payload.get("completion_tokens"))
    total_tokens = _coerce_int(payload.get("total_tokens"))
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def parse_tool_calls(raw_tool_calls: Any) -> tuple[ToolCall, ...]:
    """Parse tool-call payloads from provider responses into canonical form.

    Args:
        raw_tool_calls: Parameter value.

    Returns:
        The resulting value.
    """
    if not isinstance(raw_tool_calls, list):
        return ()
    tool_calls: list[ToolCall] = []
    for entry in raw_tool_calls:
        if not isinstance(entry, dict):
            continue
        call_id = str(entry.get("id") or "")
        name = ""
        arguments_json = ""
        function = entry.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "")
            arguments_json = str(function.get("arguments") or "")
        if name:
            if not call_id:
                call_id = f"call_{len(tool_calls) + 1}"
            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments_json=arguments_json,
                    call_id=call_id,
                )
            )
    return tuple(tool_calls)


def _coerce_int(value: Any) -> int | None:
    """Run coerce int.

    Args:
        value: Parameter value.

    Returns:
        The resulting value.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
