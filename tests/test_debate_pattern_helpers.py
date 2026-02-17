"""Focused tests for debate-pattern helper paths."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.llm import LLMChatParams, LLMMessage, LLMResponse
from design_research_agents.tools.runtime import Toolbox
from design_research_agents.workflow.implementations.debate_pattern import (
    DebatePattern,
    _merge_dependencies,
    _normalize_request_id_prefix,
    _render_prompt_template,
    _resolve_prompt_override,
    _resolve_request_id,
)


class _SequenceLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, model, params
        if not self._responses:
            raise AssertionError("No more stubbed responses available.")
        return LLMResponse(model="noop-model", text=self._responses.pop(0), provider="noop")

    def default_model(self) -> str:
        return "noop-model"


def test_debate_run_stream_emits_delta_when_streaming_enabled() -> None:
    workflow = DebatePattern(
        llm_client=_SequenceLLMClient(
            [
                "Affirmative argument",
                "Negative argument",
                json.dumps(
                    {
                        "winner": "affirmative",
                        "rationale": "Affirmative provided stronger evidence.",
                        "synthesis": "Ship a phased plan.",
                    }
                ),
            ]
        ),
        tool_runtime=Toolbox(),
        controls=RuntimeControls(max_iterations=1, streaming_enabled=True),
    )

    events = list(workflow.run_stream("Should we launch now?"))

    assert events[0].kind == "delta"
    assert events[1].kind == "completed"


def test_debate_helper_request_id_and_dependency_paths() -> None:
    assert _merge_dependencies(default_dependencies={"a": 1}, run_dependencies={"b": 2}) == {
        "a": 1,
        "b": 2,
    }
    assert _normalize_request_id_prefix(None) is None
    assert _normalize_request_id_prefix("  pref ") == "pref"
    with pytest.raises(ValueError, match="default_request_id_prefix"):
        _normalize_request_id_prefix(" ")

    assert _resolve_request_id(request_id="req-id", default_prefix=None) == "req-id"
    assert _resolve_request_id(request_id=None, default_prefix=None) is None
    assert str(_resolve_request_id(request_id=None, default_prefix="debate")).startswith("debate:")


def test_debate_prompt_helpers_cover_override_and_missing_template_keys() -> None:
    assert (
        _resolve_prompt_override(override=None, default_value="Base", field_name="field") == "Base"
    )
    assert (
        _resolve_prompt_override(override="Custom", default_value="Base", field_name="field")
        == "Custom"
    )
    assert (
        _render_prompt_template(
            template_text="Task: $task",
            variables={"task": "demo"},
            field_name="debate_prompt",
        )
        == "Task: demo"
    )
    with pytest.raises(ValueError, match="missing required variable"):
        _render_prompt_template(
            template_text="Task: $task",
            variables={},
            field_name="debate_prompt",
        )
