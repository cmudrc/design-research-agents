"""Shared helpers for trace serialization, redaction, and formatting."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

_SENSITIVE_KEYS = {
    "api_key",
    "token",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
}
_PREVIEW_MAX_LEN = 2048


def _normalize_value(value: object) -> object:
    """Normalize one value into a JSON-serializable representation.

    Args:
        value: Arbitrary value to normalize.

    Returns:
        JSON-serializable representation of ``value``.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Flatten dataclasses recursively so trace sinks only handle plain JSON-like structures.
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
        value: Arbitrary string that may be used in a filename.

    Returns:
        Filesystem-safe filename fragment.
    """
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    # Fall back to UUID so empty/fully-sanitized ids still produce unique filenames.
    return safe or uuid4().hex


def _preview(value: object, *, max_len: int = 120) -> str:
    """Build a bounded preview string for trace payload rendering.

    Args:
        value: Value to render for preview.
        max_len: Maximum preview length in characters.

    Returns:
        Single-line preview string, truncated when necessary.
    """
    if value is None:
        return "None"
    text = str(value).replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _sanitize_trace_payload(value: object, *, max_string_len: int = _PREVIEW_MAX_LEN) -> object:
    """Return one trace-safe payload with redaction and bounded string previews.

    Args:
        value: Arbitrary payload to sanitize.
        max_string_len: Maximum length for rendered string fields.

    Returns:
        Redacted and bounded payload suitable for trace attributes.
    """
    return _sanitize_trace_value(value, max_string_len=max_string_len)


def _sanitize_trace_value(value: object, *, max_string_len: int) -> object:
    """Sanitize one arbitrary value for trace storage."""
    normalized = _normalize_value(value)
    if isinstance(normalized, Mapping):
        sanitized: dict[str, object] = {}
        for key, raw_val in normalized.items():
            normalized_key = str(key)
            if normalized_key.lower() in _SENSITIVE_KEYS:
                # Redact by key-name match to avoid leaking common secret-bearing fields.
                sanitized[normalized_key] = "***REDACTED***"
                continue
            sanitized[normalized_key] = _sanitize_trace_value(raw_val, max_string_len=max_string_len)
        return sanitized
    if isinstance(normalized, list):
        return [_sanitize_trace_value(item, max_string_len=max_string_len) for item in normalized]
    if isinstance(normalized, str):
        if len(normalized) <= max_string_len:
            return normalized
        # Include original length marker so consumers can detect truncation severity.
        preview_len = max(0, max_string_len - 16)
        return f"{normalized[:preview_len]}...[truncated:{len(normalized)}]"
    return normalized
