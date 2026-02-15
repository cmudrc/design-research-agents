"""Shared helpers for injecting alternatives context into model prompts.

Agents use these helpers to route alternative/tool-option payloads into either
the system prompt or the user prompt based on one runtime input switch.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal, cast

AlternativesPromptTarget = Literal["user", "system"]


def resolve_alternatives_prompt_target(
    *,
    input_payload: Mapping[str, object],
    default_target: AlternativesPromptTarget = "user",
) -> AlternativesPromptTarget:
    """Resolve alternatives prompt target from run input.

    Supported values are ``"user"`` and ``"system"``. Invalid or missing values
    fall back to ``default_target``.

    Args:
        input_payload: Normalized run input payload mapping.
        default_target: Fallback target when no valid override is provided.

    Returns:
        Prompt target identifier for alternatives injection.
    """
    raw_target = input_payload.get("alternatives_prompt_target")
    if isinstance(raw_target, str):
        normalized_target = raw_target.strip().lower()
        if normalized_target in {"user", "system"}:
            return cast(AlternativesPromptTarget, normalized_target)
    return default_target


def append_alternatives_block(
    *,
    prompt_text: str,
    section_label: str,
    alternatives_text: str,
) -> str:
    """Append an alternatives section to a prompt when content is non-empty.

    Args:
        prompt_text: Base prompt text to append to.
        section_label: Section heading label.
        alternatives_text: Alternatives block body text.

    Returns:
        Prompt text with alternatives appended when non-empty.
    """
    normalized_alternatives = alternatives_text.strip()
    if not normalized_alternatives:
        return prompt_text
    return "\n\n".join(
        [
            prompt_text.strip(),
            f"{section_label}:\n{normalized_alternatives}",
        ]
    )


def build_alternatives_block(
    *,
    section_label: str,
    alternatives_text: str,
) -> str:
    """Build a standalone alternatives section block or return an empty string.

    Args:
        section_label: Section heading label.
        alternatives_text: Alternatives block body text.

    Returns:
        Rendered alternatives block or empty string when no content exists.
    """
    normalized_alternatives = alternatives_text.strip()
    if not normalized_alternatives:
        return ""
    return f"{section_label}:\n{normalized_alternatives}"


def build_user_prompt_alternatives_block(
    *,
    section_label: str,
    alternatives_text: str,
    target: AlternativesPromptTarget,
) -> str:
    """Build user-prompt alternatives block with optional system-routing marker.

    When ``target`` is ``"user"``, this returns the full alternatives block.
    When ``target`` is ``"system"``, this returns a short marker noting the
    alternatives were provided in the system prompt.

    Args:
        section_label: Section heading label.
        alternatives_text: Alternatives block body text.
        target: Prompt target selection for alternatives injection.

    Returns:
        User prompt alternatives block or marker string.
    """
    normalized_alternatives = alternatives_text.strip()
    if not normalized_alternatives:
        return ""
    if target == "user":
        return build_alternatives_block(
            section_label=section_label,
            alternatives_text=normalized_alternatives,
        )
    return f"{section_label}: (provided in system prompt)"


def inject_alternatives_into_prompt_pair(
    *,
    system_prompt: str,
    user_prompt: str,
    section_label: str,
    alternatives_text: str,
    target: AlternativesPromptTarget,
) -> tuple[str, str]:
    """Inject alternatives into one of a system/user prompt pair.

    Args:
        system_prompt: System prompt text to update.
        user_prompt: User prompt text to update.
        section_label: Section heading label.
        alternatives_text: Alternatives block body text.
        target: Prompt target selection for alternatives injection.

    Returns:
        Updated ``(system_prompt, user_prompt)`` tuple.
    """
    normalized_alternatives = alternatives_text.strip()
    if not normalized_alternatives:
        return system_prompt, user_prompt
    if target == "system":
        return (
            append_alternatives_block(
                prompt_text=system_prompt,
                section_label=section_label,
                alternatives_text=normalized_alternatives,
            ),
            user_prompt,
        )
    return (
        system_prompt,
        append_alternatives_block(
            prompt_text=user_prompt,
            section_label=section_label,
            alternatives_text=normalized_alternatives,
        ),
    )


def format_raw_alternatives(raw_alternatives: object) -> str:
    """Format free-form alternatives payloads into deterministic prompt text.

    Args:
        raw_alternatives: Raw alternatives payload in string, mapping, or list form.

    Returns:
        Deterministic prompt-ready alternatives text.
    """
    if raw_alternatives is None:
        return ""
    if isinstance(raw_alternatives, str):
        return raw_alternatives.strip()
    if isinstance(raw_alternatives, Mapping):
        return json.dumps(dict(raw_alternatives), sort_keys=True)
    if not isinstance(raw_alternatives, Sequence) or isinstance(raw_alternatives, (str, bytes)):
        return ""

    rendered_lines: list[str] = []
    for index, item in enumerate(raw_alternatives):
        rendered_item = _render_alternative_item(item)
        if not rendered_item:
            continue
        rendered_lines.append(f"- alternative_index: {index}\n  value: {rendered_item}")
    return "\n".join(rendered_lines)


def _render_alternative_item(item: object) -> str:
    """Render one alternatives list entry into stable text.

    Args:
        item: Alternatives item value to render.

    Returns:
        Stable string representation for prompt inclusion.
    """
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        return json.dumps(dict(item), sort_keys=True)
    if isinstance(item, (int, float, bool)):
        return str(item)
    return ""
