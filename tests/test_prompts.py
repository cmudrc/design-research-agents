"""Prompt-library tests for packaged templates and rendering behavior.

Verifies prompt discoverability, substitution behavior, and error handling.
"""

import pytest

from design_research_agents.prompts import PROMPT_NAMES, load_prompt, render_prompt


def test_all_prompts_load_from_packaged_resources() -> None:
    for prompt_name in PROMPT_NAMES:
        prompt_text = load_prompt(prompt_name)
        assert prompt_text


def test_render_prompt_substitutes_variables() -> None:
    rendered = render_prompt(
        "multi_step_step_user",
        variables={
            "task_prompt": "Do a thing",
            "step_number": 2,
            "memory_tail": "[]",
        },
    )
    assert "Task: Do a thing" in rendered
    assert "Current step: 2" in rendered
    assert "Memory tail: []" in rendered


def test_render_prompt_raises_on_missing_variables() -> None:
    with pytest.raises(ValueError):
        render_prompt(
            "multi_step_step_user",
            variables={
                "task_prompt": "Missing values",
            },
        )


def test_load_prompt_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError):
        load_prompt("unknown_prompt")
