"""Shared run-input parsing helpers for agent implementations."""

from __future__ import annotations

import json
from collections.abc import Mapping


def extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from run input with stable fallback behavior.

    Args:
        input_payload: Parameter value.

    Returns:
        The resulting value.
    """
    raw_prompt = input_payload.get(
        "prompt",
        input_payload.get("text", "Provide a concise response."),
    )
    return str(raw_prompt)


def extract_positive_int(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: int,
) -> int:
    """Extract a positive integer from run input with fallback semantics.

    Args:
        input_payload: Parameter value.
        key: Parameter value.
        default_value: Parameter value.

    Returns:
        The resulting value.
    """
    raw_value = input_payload.get(key)
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return default_value
    if isinstance(raw_value, int) and raw_value >= 1:
        return raw_value
    return default_value


def extract_boolean(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: bool,
) -> bool:
    """Extract a boolean from run input with fallback semantics.

    Args:
        input_payload: Parameter value.
        key: Parameter value.
        default_value: Parameter value.

    Returns:
        The resulting value.
    """
    raw_value = input_payload.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    return default_value


def load_json_mapping(raw_text: str) -> dict[str, object] | None:
    """Load text as a JSON mapping or return ``None`` when invalid.

    Args:
        raw_text: Parameter value.

    Returns:
        The resulting value.
    """
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    return dict(parsed)


def parse_json_mapping(raw_text: str) -> dict[str, object] | None:
    """Parse the first JSON object found in text, if any.

    Args:
        raw_text: Parameter value.

    Returns:
        The resulting value.
    """
    parsed_direct = load_json_mapping(raw_text)
    if parsed_direct is not None:
        return parsed_direct

    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_text):
        if character != "{":
            continue
        try:
            parsed_object, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_object, Mapping):
            return dict(parsed_object)
    return None
