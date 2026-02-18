from __future__ import annotations

from design_research_agents.agent.internal.json_tool_agent_helpers import (
    ToolChoice,
    build_tool_choices_text,
    coerce_tool_input,
    parse_tool_call,
    parse_tool_call_from_response,
    resolve_tool_input,
    select_tool_choice,
)
from design_research_agents.contracts.llm import LLMResponse, ToolCall


def test_parse_tool_call_helpers_cover_text_and_native_tool_call() -> None:
    native = parse_tool_call_from_response(
        LLMResponse(
            text="",
            tool_calls=(
                ToolCall(
                    name="calculator",
                    arguments_json='{"expression":"1+1"}',
                    call_id="c1",
                ),
            ),
        )
    )
    assert native == {
        "tool_name": "calculator",
        "tool_input": {"expression": "1+1"},
        "call_id": "c1",
    }

    text = parse_tool_call('prefix {"tool_name":"calculator","tool_input":{"expression":"2+2"}}')
    assert text is not None
    assert text["tool_name"] == "calculator"
    assert parse_tool_call("not-json") is None


def test_select_tool_choice_requires_valid_model_tool_name() -> None:
    choices = [
        ToolChoice(
            tool_name="calculator",
            description="math tool",
            input_schema={"type": "object"},
        ),
        ToolChoice(
            tool_name="text.word_count",
            description="text stats",
            input_schema={"type": "object"},
        ),
    ]

    selected_choice = select_tool_choice(
        parsed_tool_call={"tool_name": "calculator"},
        choices=choices,
    )
    assert selected_choice is not None
    selected, source, reason = selected_choice
    assert selected.tool_name == "calculator"
    assert source == "model"
    assert "validated" in reason

    selected2 = select_tool_choice(
        parsed_tool_call={"tool_name": "unknown"},
        choices=choices,
    )
    assert selected2 is None


def test_resolve_tool_input_precedence_and_coercion() -> None:
    choice = ToolChoice(
        tool_name="calculator",
        description="math",
        input_schema={"type": "object"},
    )
    from_model = resolve_tool_input(
        selected_choice=choice,
        parsed_tool_call={"tool_input": {"expression": "2+2"}},
        input_payload={"tool_input": {"expression": "9+9"}},
    )
    assert from_model == {"expression": "2+2"}

    from_input = resolve_tool_input(
        selected_choice=choice,
        parsed_tool_call={"tool_input": "bad"},
        input_payload={"tool_input": '{"expression":"9+9"}'},
    )
    assert from_input == {"expression": "9+9"}

    empty = resolve_tool_input(
        selected_choice=choice,
        parsed_tool_call=None,
        input_payload={},
    )
    assert isinstance(empty, dict)

    assert coerce_tool_input({"x": 1}) == {"x": 1}
    assert coerce_tool_input('{"x": 2}') == {"x": 2}
    assert coerce_tool_input("bad") is None


def test_build_tool_choices_text_renders_tool_lines() -> None:
    choices = [
        ToolChoice(
            tool_name="calculator",
            description="compute arithmetic",
            input_schema={},
        )
    ]
    rendered = build_tool_choices_text(choices=choices)
    assert "tool_name: calculator" in rendered
