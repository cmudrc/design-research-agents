"""Helpers for normalizing workflow runtime request ids and dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4


def resolve_request_id(request_id: str | None) -> str:
    """Resolve one request id into a non-empty runtime identifier.

    Args:
        request_id: Optional caller-provided request id.

    Returns:
        Stable request id value for this run.
    """
    if request_id is not None:
        normalized = request_id.strip()
        if normalized:
            return normalized
    return f"req_{uuid4().hex}"


def normalize_dependencies(dependencies: Mapping[str, object] | None) -> dict[str, object]:
    """Normalize optional dependencies mapping to a plain dictionary.

    Args:
        dependencies: Optional dependency mapping payload.

    Returns:
        Mutable dictionary for runtime usage.
    """
    if dependencies is None:
        return {}
    return dict(dependencies)


__all__ = [
    "normalize_dependencies",
    "resolve_request_id",
]
