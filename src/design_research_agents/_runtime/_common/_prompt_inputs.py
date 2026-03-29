"""Shared helpers for prompt-like inputs across workflows, agents, and patterns."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def normalize_prompt_like_input(input_payload: object) -> dict[str, object]:
    """Normalize a string prompt or one duck-typed problem object.

    String prompts preserve the historical ``{"prompt": <value>}`` payload
    exactly so existing callers and traces remain stable.

    Problem-like objects are normalized to a prompt plus optional metadata for
    workflow context builders and downstream prompt renderers.

    Args:
        input_payload: Raw caller-provided run input.

    Returns:
        Normalized prompt payload mapping.

    Raises:
        TypeError: If ``input_payload`` is neither a string prompt nor one
            supported problem-like object.
    """
    if isinstance(input_payload, str):
        return {"prompt": input_payload}

    if isinstance(input_payload, Mapping):
        raise TypeError("input must be a string prompt or a problem-like object.")

    prompt = _extract_problem_prompt(input_payload)
    if prompt is None:
        raise TypeError("input must be a string prompt or a problem-like object.")

    return {
        "prompt": prompt,
        "problem": input_payload,
        "problem_metadata": _build_problem_metadata(input_payload),
    }


def _extract_problem_prompt(problem: object) -> str | None:
    render_brief = getattr(problem, "render_brief", None)
    if callable(render_brief):
        try:
            rendered = render_brief()
        except TypeError:
            rendered = None
        except Exception:
            rendered = None
        normalized = _normalize_text(rendered)
        if normalized is not None:
            return normalized

    for attribute_name in ("statement_markdown", "brief", "prompt"):
        normalized = _normalize_text(getattr(problem, attribute_name, None))
        if normalized is not None:
            return normalized
    return None


def _build_problem_metadata(problem: object) -> dict[str, object]:
    metadata_object = getattr(problem, "metadata", None)
    metadata: dict[str, object] = {}

    for field_name in ("problem_id", "title", "kind"):
        value = _coerce_json_safe(
            _pick_first_value(
                getattr(metadata_object, field_name, None),
                getattr(problem, field_name, None),
            )
        )
        if value is not None:
            metadata[field_name] = value

    for field_name in ("candidate_kind", "family"):
        value = _coerce_json_safe(getattr(problem, field_name, None))
        if value is not None:
            metadata[field_name] = value

    return metadata


def _pick_first_value(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _coerce_json_safe(value: object) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)

    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str | int | float | bool):
        return enum_value
    return str(value)


__all__ = ["normalize_prompt_like_input"]
