from __future__ import annotations

import pytest

from design_research_agents.implementations.shared.agent_internal import (
    input_parsing,
    model_resolution,
    multi_step_loop_state,
    prompt_overrides,
    run_options,
)


def test_input_parsing_helpers_cover_fallback_and_embedded_mapping_paths() -> None:
    assert (
        input_parsing.extract_positive_int(
            input_payload={"max_steps": True},
            key="max_steps",
            default_value=3,
        )
        == 3
    )
    assert (
        input_parsing.extract_positive_int(
            input_payload={"max_steps": 0},
            key="max_steps",
            default_value=3,
        )
        == 3
    )
    assert input_parsing.load_json_mapping("[]") is None
    assert input_parsing.parse_json_mapping('prefix {"tool_name":"sum"} suffix') == {
        "tool_name": "sum"
    }


def test_prompt_template_render_raises_for_missing_variable() -> None:
    with pytest.raises(ValueError, match="missing required variable 'name'"):
        prompt_overrides.render_template_text(
            template_text="Hello $name",
            variables={},
            field_name="step_prompt",
        )


def test_model_resolution_errors_for_missing_or_non_string_default_model() -> None:
    with pytest.raises(ValueError, match="must expose default_model"):
        model_resolution.resolve_agent_model(llm_client=object())

    class _BadModelClient:
        def default_model(self) -> int:
            return 7

    with pytest.raises(ValueError, match="must return a string"):
        model_resolution.resolve_agent_model(llm_client=_BadModelClient())


def test_loop_state_coercion_rejects_invalid_container_types() -> None:
    assert multi_step_loop_state.coerce_state_records("not-a-list") == []
    assert multi_step_loop_state.coerce_string_list("not-a-list") == []
    assert multi_step_loop_state.coerce_tool_results("not-a-list") == []
    assert multi_step_loop_state.coerce_mapping("not-a-mapping") == {}


def test_normalize_input_payload_rejects_non_string_inputs() -> None:
    with pytest.raises(TypeError, match="input must be a string prompt"):
        run_options.normalize_input_payload(123)  # type: ignore[arg-type]
