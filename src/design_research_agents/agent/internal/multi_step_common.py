"""Shared helpers for multi-step agent loop behavior."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents.agent.internal.prompt_overrides import render_template_text


def build_continue_prompt(
    *,
    prompt: str,
    memory: Sequence[Mapping[str, object]],
    step_number: int,
    prompt_template: str,
    memory_tail_items: int = 6,
) -> str:
    """Build the continuation-decision prompt from task context and memory."""
    memory_preview = json.dumps(list(memory)[-memory_tail_items:], sort_keys=True)
    return render_template_text(
        template_text=prompt_template,
        variables={
            "step_number": step_number,
            "task_prompt": prompt,
            "memory_tail": memory_preview,
        },
        field_name="continuation_user_prompt_template",
    )


def build_step_prompt(
    *,
    prompt: str,
    memory: Sequence[Mapping[str, object]],
    step_number: int,
    prompt_template: str,
    memory_tail_items: int = 8,
) -> str:
    """Build one action-step prompt from task context and memory."""
    memory_preview = json.dumps(list(memory)[-memory_tail_items:], sort_keys=True)
    return render_template_text(
        template_text=prompt_template,
        variables={
            "task_prompt": prompt,
            "step_number": step_number,
            "memory_tail": memory_preview,
        },
        field_name="step_user_prompt_template",
    )


def extract_continuation_thought(parsed: Mapping[str, object]) -> str:
    """Extract normalized continuation thought text from model output."""
    thought = parsed.get("thought")
    if thought is not None:
        return str(thought)
    return "model decision"


def fallback_should_continue(
    *,
    memory: Sequence[Mapping[str, object]],
    step_index: int,
    max_steps: int,
) -> bool:
    """Fallback continuation policy used when model output is invalid JSON."""
    if step_index >= max_steps:
        return False

    # On parse failure, guarantee one first step, then stop by default.
    if step_index == 0:
        return True

    # If the last observation failed, stop.
    for entry in reversed(memory):
        if entry.get("kind") != "observation":
            continue
        if entry.get("success") is False:
            return False
        break

    return False


def has_observation(memory: Sequence[Mapping[str, object]]) -> bool:
    """Return whether memory includes at least one observation entry."""
    return any(entry.get("kind") == "observation" for entry in memory)
