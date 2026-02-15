from __future__ import annotations

from design_research_agents.agent.implementations.single_step_direct_llm_agent import (
    _coerce_provider_options,
    _extract_max_tokens,
    _extract_messages,
    _extract_response_schema,
    _extract_system_prompt,
    _extract_temperature,
    _inject_alternatives_into_system_message,
    _inject_alternatives_into_user_message,
    _merge_provider_options,
    _normalize_messages,
)
from design_research_agents.contracts.llm import LLMMessage


def test_normalize_messages_filters_invalid_entries() -> None:
    normalized = _normalize_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "rules", "name": "   planner  "},
            {"role": "bad-role", "content": "x"},
            {"role": "assistant", "content": 1},
            "invalid",
        ]
    )
    assert normalized == [
        LLMMessage(role="user", content="hi", name=None),
        LLMMessage(role="system", content="rules", name="planner"),
    ]


def test_extract_system_prompt_and_messages_paths() -> None:
    assert _extract_system_prompt(input_payload={}, default_system_prompt=None) is None
    assert (
        _extract_system_prompt(
            input_payload={"system_prompt": "  custom  "},
            default_system_prompt="fallback",
        )
        == "custom"
    )

    messages, source = _extract_messages(
        input_payload={"prompt": "hello", "system_prompt": "sys"},
        default_system_prompt=None,
    )
    assert source == "prompt"
    assert messages[0].role == "system"
    assert messages[-1].content == "hello"


def test_extract_messages_prefers_explicit_messages_and_alternatives() -> None:
    messages, source = _extract_messages(
        input_payload={
            "messages": [
                {"role": "user", "content": "pick"},
            ],
            "alternatives": ["a", "b"],
            "alternatives_prompt_target": "system",
        },
        default_system_prompt=None,
    )
    assert source == "messages"
    assert messages[0].role == "system"
    assert "Available alternatives" in messages[0].content


def test_inject_alternatives_into_user_and_system_messages() -> None:
    system_injected = _inject_alternatives_into_system_message(
        messages=[LLMMessage(role="user", content="hi")],
        alternatives_text="- option: one",
    )
    assert system_injected[0].role == "system"
    assert "Available alternatives" in system_injected[0].content

    user_injected = _inject_alternatives_into_user_message(
        messages=[LLMMessage(role="assistant", content="x")],
        alternatives_text="- option: one",
    )
    assert user_injected[-1].role == "user"
    assert "Available alternatives" in user_injected[-1].content


def test_extract_numeric_and_provider_options_helpers() -> None:
    payload = {"temperature": "0.3", "max_tokens": "64", "provider_options": {"seed": 9}}
    assert _extract_temperature(input_payload=payload, default_value=0.7) == 0.3
    assert _extract_temperature(input_payload={"temperature": "bad"}, default_value=0.7) == 0.7
    assert _extract_max_tokens(input_payload=payload, default_value=10) == 64
    assert _extract_max_tokens(input_payload={"max_tokens": 0}, default_value=10) == 10
    assert _extract_response_schema({"response_schema": {"type": "object"}}) == {"type": "object"}
    assert _extract_response_schema({"response_schema": "x"}) is None

    assert _coerce_provider_options({"a": 1, 2: 3}) == {"a": 1}
    assert _coerce_provider_options("bad") == {}
    merged = _merge_provider_options(
        default_provider_options={"x": 1},
        raw_provider_options={"y": 2},
    )
    assert merged == {"x": 1, "y": 2}
