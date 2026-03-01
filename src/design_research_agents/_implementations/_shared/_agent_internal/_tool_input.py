"""Shared helpers for tool-aware input resolution in agent implementations."""

from __future__ import annotations

import re
from collections.abc import Mapping

from ._input_parsing import extract_prompt

CALCULATOR_TOOL_NAMES = frozenset({"calculator"})
TEXT_WORD_COUNT_TOOL_NAMES = frozenset({"text.word_count"})
FS_READ_TEXT_TOOL_NAMES = frozenset({"fs.read_text"})
_FILE_PATH_TOKEN = re.compile(r"(?<![\w./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9_.-]+)(?![\w./-])")


def infer_expression(*, input_payload: Mapping[str, object], prompt: str) -> str:
    """Infer arithmetic expression from payload fields and prompt text.

    Args:
        input_payload: Value supplied for ``input_payload``.
        prompt: Value supplied for ``prompt``.

    Returns:
        Result produced by this call.
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


def infer_file_path(*, input_payload: Mapping[str, object], prompt: str) -> str | None:
    """Infer a likely file path from explicit fields or prompt text.

    Args:
        input_payload: Value supplied for ``input_payload``.
        prompt: Value supplied for ``prompt``.

    Returns:
        Best-effort file path when one can be inferred, otherwise ``None``.
    """
    for field_name in ("path", "file_path", "filename"):
        explicit_value = input_payload.get(field_name)
        if explicit_value is None:
            continue
        explicit_path = str(explicit_value).strip()
        if _looks_like_file_path(explicit_path):
            return explicit_path

    for match in re.finditer(r"`([^`\n]+)`", prompt):
        candidate = match.group(1).strip()
        if _looks_like_file_path(candidate):
            return candidate

    prompt_match = _FILE_PATH_TOKEN.search(prompt)
    if prompt_match is not None:
        return prompt_match.group(1)

    return None


def resolve_known_tool_input(
    *,
    tool_name: str,
    input_payload: Mapping[str, object],
) -> dict[str, object] | None:
    """Build heuristic input payloads for known core tool families.

    Args:
        tool_name: Value supplied for ``tool_name``.
        input_payload: Value supplied for ``input_payload``.

    Returns:
        Result produced by this call.
    """
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
        return {"text": extract_prompt(input_payload)}

    if tool_name in FS_READ_TEXT_TOOL_NAMES:
        inferred_path = infer_file_path(
            input_payload=input_payload,
            prompt=extract_prompt(input_payload),
        )
        if inferred_path is not None:
            return {"path": inferred_path}

    return None


def _looks_like_file_path(value: str) -> bool:
    """Return whether a string looks like a file path with an extension."""
    stripped = value.strip()
    if not stripped or "://" in stripped or "//" in stripped:
        return False
    candidate = stripped[1:] if stripped.startswith("/") else stripped
    return _FILE_PATH_TOKEN.fullmatch(candidate) is not None


__all__ = [
    "extract_prompt",
    "infer_file_path",
    "resolve_known_tool_input",
]
