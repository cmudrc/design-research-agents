"""Tests for shared internal agent helpers used across implementations."""

from __future__ import annotations

from design_research_agents.agent.internal.input_parsing import (
    extract_boolean,
    extract_positive_int,
    extract_prompt,
    load_json_mapping,
    parse_json_mapping,
)
from design_research_agents.agent.internal.multi_step_common import (
    fallback_should_continue,
    has_observation,
)
from design_research_agents.agent.internal.result_builders import build_failure_result


def test_extract_prompt_uses_prompt_then_text_then_default() -> None:
    assert extract_prompt({"prompt": "direct"}) == "direct"
    assert extract_prompt({"text": "fallback text"}) == "fallback text"
    assert extract_prompt({}) == "Provide a concise response."


def test_extract_positive_int_and_boolean_respect_defaults() -> None:
    payload = {"ok_int": 3, "bad_int": 0, "bool_like": True, "flag": False}
    assert extract_positive_int(input_payload=payload, key="ok_int", default_value=1) == 3
    assert extract_positive_int(input_payload=payload, key="bad_int", default_value=2) == 2
    assert extract_positive_int(input_payload=payload, key="bool_like", default_value=4) == 4
    assert extract_positive_int(input_payload=payload, key="missing", default_value=5) == 5
    assert extract_boolean(input_payload=payload, key="flag", default_value=True) is False
    assert extract_boolean(input_payload=payload, key="missing", default_value=True) is True


def test_json_mapping_helpers_parse_direct_and_embedded_objects() -> None:
    assert load_json_mapping('{"a": 1}') == {"a": 1}
    assert load_json_mapping("[]") is None
    assert load_json_mapping("not-json") is None
    assert parse_json_mapping("prefix {'broken': 1}") is None
    assert parse_json_mapping('prefix {"a": 1, "b": {"c": 2}} suffix') == {
        "a": 1,
        "b": {"c": 2},
    }


def test_multi_step_fallback_policy_behaves_deterministically() -> None:
    memory = [{"kind": "task", "prompt": "hello"}]
    assert fallback_should_continue(memory=memory, step_index=0, max_steps=3) is True
    assert fallback_should_continue(memory=memory, step_index=1, max_steps=3) is False
    failed_memory = [
        *memory,
        {"kind": "observation", "success": False},
    ]
    assert fallback_should_continue(memory=failed_memory, step_index=1, max_steps=3) is False
    assert has_observation(memory) is False
    assert has_observation(failed_memory) is True


def test_build_failure_result_merges_metadata_and_output() -> None:
    result = build_failure_result(
        error="boom",
        model_response=None,
        tool_results=[],
        request_id="req-1",
        dependencies={"k": "v"},
        metadata={"stage": "test"},
        output={"extra": 1},
    )
    assert result.success is False
    assert result.output["error"] == "boom"
    assert result.output["extra"] == 1
    assert result.metadata["request_id"] == "req-1"
    assert result.metadata["dependency_keys"] == ["k"]
    assert result.metadata["stage"] == "test"
