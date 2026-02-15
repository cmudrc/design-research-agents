"""Shared helpers for tool-aware input resolution in agent implementations."""

from __future__ import annotations

import re
from collections.abc import Mapping

DEFAULT_FALLBACK_TOOL_NAME = "text.word_count"
CALCULATOR_TOOL_NAMES = frozenset({"calculator"})
TEXT_WORD_COUNT_TOOL_NAMES = frozenset({"text.word_count"})


def extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from run input with stable fallback behavior."""
    raw_prompt = input_payload.get(
        "prompt", input_payload.get("text", "Provide a concise response.")
    )
    return str(raw_prompt)


def infer_expression(*, input_payload: Mapping[str, object], prompt: str) -> str:
    """Infer arithmetic expression from payload fields and prompt text."""
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


def resolve_known_tool_input(
    *,
    tool_name: str,
    input_payload: Mapping[str, object],
    text_fallback: str,
) -> dict[str, object] | None:
    """Build heuristic input payloads for known core tool families."""
    if tool_name in CALCULATOR_TOOL_NAMES:
        return {
            "expression": infer_expression(
                input_payload=input_payload,
                prompt=extract_prompt(input_payload),
            )
        }

    if tool_name in TEXT_WORD_COUNT_TOOL_NAMES:
        analysis_text = input_payload.get("analysis_text")
        if analysis_text is not None:
            return {"text": str(analysis_text)}
        return {"text": text_fallback}

    return None


__all__ = [
    "DEFAULT_FALLBACK_TOOL_NAME",
    "extract_prompt",
    "resolve_known_tool_input",
]
