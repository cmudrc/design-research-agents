"""Helpers for normalizing per-run input, identifiers, and dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4


def normalize_input_payload(input_payload: str) -> dict[str, object]:
    """Normalize one run input payload into a plain dictionary.

    Prompts are stored as ``{"prompt": <value>}`` for downstream helpers.
    """
    if not isinstance(input_payload, str):
        raise TypeError("input must be a string prompt.")
    return {"prompt": input_payload}


def resolve_request_id(request_id: str | None) -> str:
    """Return a stable request id for one run.

    Empty, whitespace-only, and non-string ids are rejected by callers before
    this helper is called, so this function only handles ``None`` and strings.
    """
    if request_id is not None:
        normalized_request_id = request_id.strip()
        if normalized_request_id:
            return normalized_request_id
    return f"req_{uuid4().hex}"


def normalize_dependencies(
    dependencies: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize optional dependency payload into a plain dictionary."""
    if dependencies is None:
        return {}
    return dict(dependencies)
