"""Shared helpers for multi-step agent loop behavior."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents._implementations._shared._agent_internal._prompt_overrides import (
    render_template_text,
)


def build_continue_prompt(
    *,
    prompt: str,
    memory: Sequence[Mapping[str, object]],
    step_number: int,
    prompt_template: str,
    memory_tail_items: int = 6,
    retrieved_context: str = "",
) -> str:
    """Build the continuation-decision prompt from task context and memory.

    Args:
        prompt: Value supplied for ``prompt``.
        memory: Value supplied for ``memory``.
        step_number: Value supplied for ``step_number``.
        prompt_template: Value supplied for ``prompt_template``.
        memory_tail_items: Value supplied for ``memory_tail_items``.
        retrieved_context: Optional retrieved-context block injected into prompt.

    Returns:
        Result produced by this call.
    """
    memory_preview = json.dumps(list(memory)[-memory_tail_items:], sort_keys=True)
    return render_template_text(
        template_text=prompt_template,
        variables={
            "step_number": step_number,
            "task_prompt": prompt,
            "memory_tail": memory_preview,
            "retrieved_context": retrieved_context or "(none)",
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
    retrieved_context: str = "",
) -> str:
    """Build one action-step prompt from task context and memory.

    Args:
        prompt: Value supplied for ``prompt``.
        memory: Value supplied for ``memory``.
        step_number: Value supplied for ``step_number``.
        prompt_template: Value supplied for ``prompt_template``.
        memory_tail_items: Value supplied for ``memory_tail_items``.
        retrieved_context: Optional retrieved-context block injected into prompt.

    Returns:
        Result produced by this call.
    """
    memory_preview = json.dumps(list(memory)[-memory_tail_items:], sort_keys=True)
    return render_template_text(
        template_text=prompt_template,
        variables={
            "task_prompt": prompt,
            "step_number": step_number,
            "memory_tail": memory_preview,
            "retrieved_context": retrieved_context or "(none)",
        },
        field_name="step_user_prompt_template",
    )


def extract_continuation_thought(parsed: Mapping[str, object]) -> str:
    """Extract normalized continuation thought text from model output.

    Args:
        parsed: Value supplied for ``parsed``.

    Returns:
        Result produced by this call.
    """
    thought = parsed.get("thought")
    if thought is not None:
        return str(thought)
    return "model decision"


def has_observation(memory: Sequence[Mapping[str, object]]) -> bool:
    """Return whether memory includes at least one observation entry.

    Args:
        memory: Value supplied for ``memory``.

    Returns:
        Result produced by this call.
    """
    return any(entry.get("kind") == "observation" for entry in memory)
