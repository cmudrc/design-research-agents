"""Helpers for normalizing per-run input, identifiers, and dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

AgentInput = Mapping[str, object] | str


def normalize_input_payload(input_payload: AgentInput) -> dict[str, object]:
    """Normalize one run input payload into a plain dictionary.

    Plain string inputs are treated as shorthand for ``{"prompt": <value>}``.
    """
    if isinstance(input_payload, str):
        return {"prompt": input_payload}
    if isinstance(input_payload, Mapping):
        return dict(input_payload)
    raise TypeError("input must be either a string prompt or a mapping payload.")


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
