"""Loop-state normalization utilities shared across workflow patterns."""

from __future__ import annotations

from collections.abc import Mapping


def normalize_mapping(raw_value: object) -> dict[str, object]:
    """Normalize an arbitrary object into a mutable mapping copy.

    Args:
        raw_value: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if not isinstance(raw_value, Mapping):
        return {}
    return dict(raw_value)


def normalize_mapping_records(raw_value: object) -> list[dict[str, object]]:
    """Normalize an arbitrary object into list-of-mapping records.

    Args:
        raw_value: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if not isinstance(raw_value, list):
        return []
    normalized: list[dict[str, object]] = []
    for entry in raw_value:
        if isinstance(entry, Mapping):
            normalized.append(dict(entry))
    return normalized


def parse_loop_iteration(raw_value: object, *, error_prefix: str) -> int:
    """Parse one loop-iteration value from loop metadata.

    Args:
        raw_value: Input value for this parameter.
        error_prefix: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when validation or execution fails.
    """
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, str) and raw_value.strip():
        return int(raw_value)
    raise ValueError(f"{error_prefix} must be an integer.")


__all__ = [
    "normalize_mapping",
    "normalize_mapping_records",
    "parse_loop_iteration",
]
