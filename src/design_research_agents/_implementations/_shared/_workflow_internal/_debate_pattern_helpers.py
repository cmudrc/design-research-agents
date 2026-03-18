"""Shared helpers for debate-pattern prompt rendering and normalization."""

from __future__ import annotations

import json

from design_research_agents._runtime._patterns import render_prompt_template


def render_judge_prompt(
    *,
    prompt_template: str,
    task_prompt: str,
    rounds: list[dict[str, object]],
) -> str:
    """Render judge prompt from task prompt and normalized rounds."""
    return render_prompt_template(
        template_text=prompt_template,
        variables={
            "task_prompt": task_prompt,
            "debate_rounds_json": json.dumps(rounds, ensure_ascii=True, sort_keys=True),
        },
        field_name="judge_user_prompt_template",
    )


def normalize_optional_text(value: object) -> str | None:
    """Normalize optional text value."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


__all__ = ["normalize_optional_text", "render_judge_prompt"]
