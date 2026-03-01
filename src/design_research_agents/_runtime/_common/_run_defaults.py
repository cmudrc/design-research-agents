"""Canonical normalization helpers for workflow and pattern run defaults."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4


def merge_dependencies(
    *,
    default_dependencies: Mapping[str, object],
    run_dependencies: Mapping[str, object] | None,
) -> dict[str, object]:
    """Merge constructor-level and per-run dependency dictionaries."""
    merged = dict(default_dependencies)
    if run_dependencies is not None:
        merged.update(run_dependencies)
    return merged


def normalize_request_id_prefix(default_request_id_prefix: str | None) -> str | None:
    """Validate one optional request-id prefix value."""
    if default_request_id_prefix is None:
        return None
    normalized = default_request_id_prefix.strip()
    if not normalized:
        raise ValueError("default_request_id_prefix must be non-empty when provided.")
    return normalized


def resolve_request_id_with_prefix(
    *,
    request_id: str | None,
    default_prefix: str | None,
) -> str | None:
    """Resolve one request-id with optional prefix fallback."""
    if request_id is not None and request_id.strip():
        return request_id
    if default_prefix is None:
        return request_id
    return f"{default_prefix}:{uuid4().hex}"


__all__ = [
    "merge_dependencies",
    "normalize_request_id_prefix",
    "resolve_request_id_with_prefix",
]
