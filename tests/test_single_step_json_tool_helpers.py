from __future__ import annotations

from design_research_agents.agent.implementations.single_step_json_tool_calling_agent import (
    _build_tool_choices_text,
    _coerce_tool_input,
    _fallback_select_tool_choice,
    _looks_like_arithmetic_request,
    _looks_like_arithmetic_tool,
    _looks_like_text_analysis_request,
    _looks_like_text_tool,
    _parse_tool_call,
    _parse_tool_call_from_response,
    _resolve_tool_input,
    _select_tool_choice,
    _tokenize,
    _ToolChoice,
)
from design_research_agents.contracts.llm import LLMResponse, ToolCall


def test_parse_tool_call_helpers_cover_text_and_native_tool_call() -> None:
    native = _parse_tool_call_from_response(
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

    text = _parse_tool_call('prefix {"tool_name":"calculator","tool_input":{"expression":"2+2"}}')
    assert text is not None
    assert text["tool_name"] == "calculator"
    assert _parse_tool_call("not-json") is None


def test_select_tool_choice_prefers_valid_model_and_falls_back() -> None:
    choices = [
        _ToolChoice(
            tool_name="calculator",
            description="math tool",
            input_schema={"type": "object"},
        ),
        _ToolChoice(
            tool_name="text.word_count",
            description="text stats",
            input_schema={"type": "object"},
        ),
    ]

    selected, source, reason = _select_tool_choice(
        parsed_tool_call={"tool_name": "calculator"},
        prompt="count words",
        choices=choices,
    )
    assert selected.tool_name == "calculator"
    assert source == "model"
    assert "validated" in reason

    selected2, source2, reason2 = _select_tool_choice(
        parsed_tool_call={"tool_name": "unknown"},
        prompt="compute 3*7",
        choices=choices,
    )
    assert selected2.tool_name == "calculator"
    assert source2 == "fallback"
    assert isinstance(reason2, str)


def test_resolve_tool_input_precedence_and_coercion() -> None:
    choice = _ToolChoice(
        tool_name="calculator",
        description="math",
        input_schema={"type": "object"},
    )
    from_model = _resolve_tool_input(
        selected_choice=choice,
        parsed_tool_call={"tool_input": {"expression": "2+2"}},
        input_payload={"tool_input": {"expression": "9+9"}},
        llm_response_text="ignored",
    )
    assert from_model == {"expression": "2+2"}

    from_input = _resolve_tool_input(
        selected_choice=choice,
        parsed_tool_call={"tool_input": "bad"},
        input_payload={"tool_input": '{"expression":"9+9"}'},
        llm_response_text="ignored",
    )
    assert from_input == {"expression": "9+9"}

    empty = _resolve_tool_input(
        selected_choice=choice,
        parsed_tool_call=None,
        input_payload={},
        llm_response_text="n/a",
    )
    assert isinstance(empty, dict)

    assert _coerce_tool_input({"x": 1}) == {"x": 1}
    assert _coerce_tool_input('{"x": 2}') == {"x": 2}
    assert _coerce_tool_input("bad") is None


def test_fallback_heuristics_and_tokenization() -> None:
    choices = [
        _ToolChoice(tool_name="calculator", description="compute arithmetic", input_schema={}),
        _ToolChoice(tool_name="text.word_count", description="count words", input_schema={}),
    ]
    selected_math, reason_math = _fallback_select_tool_choice(
        prompt="Please calculate 12*3",
        choices=choices,
    )
    assert selected_math.tool_name == "calculator"
    assert "math-signal" in reason_math or "token-overlap" in reason_math

    selected_text, reason_text = _fallback_select_tool_choice(
        prompt="count characters in this text",
        choices=choices,
    )
    assert selected_text.tool_name == "text.word_count"
    assert reason_text

    assert _tokenize("A-b_c 123!") == {"b_c", "123"}
    assert _looks_like_arithmetic_request("solve 3 + 4") is True
    assert _looks_like_text_analysis_request("word count this paragraph") is True
    assert _looks_like_arithmetic_tool("best calculator") is True
    assert _looks_like_text_tool("summarize text") is True

    rendered = _build_tool_choices_text(choices=choices)
    assert "tool_name: calculator" in rendered
