"""Shared helpers for constructor-time prompt override validation/rendering."""

from __future__ import annotations

from collections.abc import Mapping
from string import Template

from design_research_agents._prompts import load_prompt


def resolve_prompt_text(
    *,
    override: str | None,
    default_prompt_name: str,
    field_name: str,
) -> str:
    """Resolve prompt text from optional override or packaged default.

    Args:
        override: Input value for this parameter.
        default_prompt_name: Input value for this parameter.
        field_name: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if override is None:
        return load_prompt(default_prompt_name)
    return validate_prompt_text(value=override, field_name=field_name)


def validate_prompt_text(*, value: str, field_name: str) -> str:
    """Validate one prompt/template override and return normalized text.

    Args:
        value: Input value for this parameter.
        field_name: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string when provided.")
    return normalized


def render_template_text(
    *,
    template_text: str,
    variables: Mapping[str, object],
    field_name: str,
) -> str:
    """Render template text with strict missing-key validation.

    Args:
        template_text: Input value for this parameter.
        variables: Input value for this parameter.
        field_name: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    normalized_template = validate_prompt_text(value=template_text, field_name=field_name)
    normalized_variables = {key: str(value) for key, value in variables.items()}
    template = Template(normalized_template)
    try:
        return template.substitute(normalized_variables)
    except KeyError as exc:
        missing_key = exc.args[0] if exc.args else "unknown"
        raise ValueError(f"{field_name} is missing required variable '{missing_key}'.") from exc
