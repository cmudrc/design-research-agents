"""Shared helpers for core tool implementations."""

from __future__ import annotations

from collections.abc import Mapping


def get_str(input_dict: Mapping[str, object], key: str, *, default: str = "") -> str:
    """Run get str.

    Args:
        input_dict: Parameter value.
        key: Parameter value.
        default: Parameter value.

    Returns:
        The resulting value.
    """
    value = input_dict.get(key, default)
    return str(value)


def get_int(input_dict: Mapping[str, object], key: str, *, default: int) -> int:
    """Run get int.

    Args:
        input_dict: Parameter value.
        key: Parameter value.
        default: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    value = input_dict.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"'{key}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    raise ValueError(f"'{key}' must be an integer.")


def get_bool(input_dict: Mapping[str, object], key: str, *, default: bool = False) -> bool:
    """Run get bool.

    Args:
        input_dict: Parameter value.
        key: Parameter value.
        default: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    value = input_dict.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"'{key}' must be a boolean.")
