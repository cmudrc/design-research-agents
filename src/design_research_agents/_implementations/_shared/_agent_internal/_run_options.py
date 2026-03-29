"""Helpers for normalizing per-run input, identifiers, and dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

from design_research_agents._runtime._common._prompt_inputs import (
    normalize_prompt_like_input,
)


def normalize_input_payload(input_payload: str | object) -> dict[str, object]:
    """Normalize one run input payload into a plain dictionary.

    String prompts preserve the historical ``{"prompt": <value>}`` shape. When
    callers supply one problem-like object, the payload also carries ``problem``
    and ``problem_metadata`` fields for downstream prompt builders.

    Args:
        input_payload: Raw prompt text or problem-like object.

    Returns:
        Normalized input payload mapping.
    """
    return normalize_prompt_like_input(input_payload)


def resolve_request_id(request_id: str | None) -> str:
    """Return a stable request id for one run.

    Empty, whitespace-only, and non-string ids are rejected by callers before
    this helper is called, so this function only handles ``None`` and strings.

    Args:
        request_id: Optional caller-provided request id.

    Returns:
        Non-empty request id suitable for tracing.
    """
    if request_id is not None:
        normalized_request_id = request_id.strip()
        if normalized_request_id:
            return normalized_request_id
    return f"req_{uuid4().hex}"


def normalize_dependencies(
    dependencies: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize optional dependency payload into a plain dictionary.

    Args:
        dependencies: Optional dependency mapping.

    Returns:
        Normalized dependency mapping.
    """
    if dependencies is None:
        return {}
    return dict(dependencies)
