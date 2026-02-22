"""Shared helpers for trace serialization and formatting."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4


def _normalize_value(value: object) -> object:
    """Normalize one value into a JSON-serializable representation.

    Args:
        value: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_value(dataclasses.asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_filename(value: str) -> str:
    """Normalize one arbitrary string for safe trace-file naming.

    Args:
        value: Input value for this parameter.

    Returns:
        Computed return value.
    """
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe or uuid4().hex


def _preview(value: object, *, max_len: int = 120) -> str:
    """Build a bounded preview string for trace payload rendering.

    Args:
        value: Input value for this parameter.
        max_len: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if value is None:
        return "None"
    text = str(value).replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."
