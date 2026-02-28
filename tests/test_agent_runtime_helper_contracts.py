from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from design_research_agents._contracts._llm import LLMMessage, LLMRequest, LLMResponse, ToolCall
from design_research_agents._contracts._tools import ToolSpec
from design_research_agents._implementations._shared._agent_internal._direct_llm_agent_helpers import (
    build_success_result,
    extract_max_tokens,
    extract_messages,
    extract_response_schema,
    extract_temperature,
    generate_response,
    merge_provider_options,
)
from design_research_agents._implementations._shared._agent_internal._json_tool_agent_helpers import (
    ToolChoice,
    build_tool_call_prompt,
    build_tool_choices_text,
    clone_tool_choice,
    coerce_tool_input,
    extract_tool_choices,
    parse_tool_call,
    parse_tool_call_from_response,
    request_tool_call_response,
    resolve_allowed_tool_names,
    resolve_tool_input,
    select_tool_choice,
    tool_call_response_schema,
)
from design_research_agents._implementations._shared._agent_internal._prompt_alternatives import (
    append_alternatives_block,
    build_alternatives_block,
    build_user_prompt_alternatives_block,
    format_raw_alternatives,
    inject_alternatives_into_prompt_pair,
    normalize_alternatives_prompt_target,
    resolve_alternatives_prompt_target,
)
from design_research_agents._implementations._shared._agent_internal._tool_input import (
    infer_expression,
    infer_file_path,
    resolve_known_tool_input,
)

pytestmark = pytest.mark.contract


class _GenerateClient:
    def default_model(self) -> str:
        return "fallback"

    def close(self) -> None:
        return None

    def __enter__(self) -> _GenerateClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="generated", model=request.model, provider="test")


class _ChatClient:
    def default_model(self) -> str:
        return "chat-default"

    def close(self) -> None:
        return None

    def __enter__(self) -> _ChatClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: object,
    ) -> LLMResponse:
        del messages, params
        return LLMResponse(text="chat", model=model, provider="test")


def test_generate_response_uses_generate_chat_or_raises() -> None:
    request = LLMRequest(messages=[LLMMessage(role="user", content="hello")], model="m")

    generated = generate_response(_GenerateClient(), request)
    assert generated.text == "generated"

    chat = generate_response(_ChatClient(), request)
    assert chat.text == "chat"
    assert chat.model == "m"

    with pytest.raises(AttributeError, match=r"generate\(\) or compatible chat"):
        generate_response(SimpleNamespace(default_model=lambda: "x"), request)


def test_direct_llm_message_and_parameter_extractors_cover_fallbacks() -> None:
    messages, source = extract_messages(
        input_payload={
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "invalid", "content": "x"},
            ],
            "alternatives": [{"name": "a"}],
            "alternatives_prompt_target": "system",
        },
        default_system_prompt="default",
    )
    assert source == "messages"
    assert any(message.role == "system" and "Available alternatives" in message.content for message in messages)

    prompt_messages, prompt_source = extract_messages(
        input_payload={"prompt": "task", "alternatives": ["a", 1]},
        default_system_prompt="system",
    )
    assert prompt_source == "prompt"
    assert prompt_messages[0].role == "system"
    assert prompt_messages[-1].role == "user"

    assert extract_response_schema({"response_schema": {"type": "object", 1: "x"}}) == {"type": "object"}
    assert extract_response_schema({"response_schema": "bad"}) is None

    assert extract_temperature(input_payload={"temperature": "0.2"}, default_value=0.7) == 0.2
    assert extract_temperature(input_payload={"temperature": "bad"}, default_value=0.7) == 0.7
    assert extract_max_tokens(input_payload={"max_tokens": "32"}, default_value=16) == 32
    assert extract_max_tokens(input_payload={"max_tokens": "-1"}, default_value=16) == 16

    merged = merge_provider_options(
        default_provider_options={"a": 1},
        raw_provider_options={"b": 2, 3: "skip"},
    )
    assert merged == {"a": 1, "b": 2}


def test_build_success_result_includes_llm_call_metadata() -> None:
    request = LLMRequest(
        messages=[LLMMessage(role="user", content="hello")],
        model="m",
        temperature=0.1,
        max_tokens=32,
        response_schema={"type": "object"},
        provider_options={"region": "us"},
    )
    response = LLMResponse(text="done", model="m", provider="test")

    result = build_success_result(
        llm_response=response,
        request_id="req-1",
        dependencies={"upstream": 1},
        message_source="prompt",
        message_count=1,
        llm_request=request,
    )

    assert result.success is True
    assert result.metadata["request_id"] == "req-1"
    assert result.metadata["llm_call"]["response_schema_supplied"] is True


def test_prompt_alternatives_helpers_cover_targets_and_rendering() -> None:
    assert normalize_alternatives_prompt_target(" user ") == "user"
    with pytest.raises(ValueError, match="alternatives_prompt_target"):
        normalize_alternatives_prompt_target("bad")

    assert resolve_alternatives_prompt_target(input_payload={"alternatives_prompt_target": "bad"}) == "user"
    assert resolve_alternatives_prompt_target(input_payload={"alternatives_prompt_target": "system"}) == "system"

    assert (
        append_alternatives_block(
            prompt_text="base",
            section_label="Alt",
            alternatives_text="",
        )
        == "base"
    )
    assert "Alt:" in append_alternatives_block(
        prompt_text="base",
        section_label="Alt",
        alternatives_text="one",
    )

    assert build_alternatives_block(section_label="Alt", alternatives_text="") == ""
    assert "provided in system prompt" in build_user_prompt_alternatives_block(
        section_label="Alt",
        alternatives_text="x",
        target="system",
    )

    system_prompt, user_prompt = inject_alternatives_into_prompt_pair(
        system_prompt="system",
        user_prompt="user",
        section_label="Alt",
        alternatives_text="choice",
        target="user",
    )
    assert system_prompt == "system"
    assert "Alt:" in user_prompt

    formatted = format_raw_alternatives(["a", {"x": 1}, 2, True, object()])
    assert "alternative_index" in formatted
    assert '"x": 1' in formatted


def test_tool_input_helpers_cover_known_tool_resolution() -> None:
    assert infer_expression(input_payload={"expression": "1+1"}, prompt="ignored") == "1+1"
    assert infer_expression(input_payload={"text": "2*3"}, prompt="ignored") == "2*3"
    assert infer_expression(input_payload={}, prompt="Compute 4 + 5") == "4 + 5"
    assert infer_expression(input_payload={}, prompt="no math") == "no math"
    assert infer_file_path(input_payload={}, prompt="Read README.md next.") == "README.md"
    assert infer_file_path(input_payload={"path": "docs/api.rst"}, prompt="ignored") == "docs/api.rst"
    assert infer_file_path(input_payload={}, prompt="No file mentioned here.") is None

    calc_input = resolve_known_tool_input(tool_name="calculator", input_payload={"prompt": "7*8"})
    assert calc_input == {"expression": "7*8"}

    word_count = resolve_known_tool_input(
        tool_name="text.word_count",
        input_payload={"analysis_text": "hello world"},
    )
    assert word_count == {"text": "hello world"}
    assert resolve_known_tool_input(tool_name="fs.read_text", input_payload={"prompt": "Read README.md"}) == {
        "path": "README.md"
    }
    assert resolve_known_tool_input(tool_name="unknown", input_payload={"prompt": "x"}) is None


def test_json_tool_helpers_cover_choice_resolution_and_tool_input_precedence() -> None:
    specs = {
        "calculator": ToolSpec(
            name="calculator",
            description="calc",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
        "text.word_count": ToolSpec(
            name="text.word_count",
            description="count",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
    }

    choices = extract_tool_choices(tool_specs=specs)
    assert len(choices) == 2

    with pytest.raises(ValueError, match="at least one tool"):
        extract_tool_choices(tool_specs=specs, allowed_tool_names=["missing"])

    allowed = resolve_allowed_tool_names(runtime_specs=specs, allowed_tools=[" calculator ", "bad"])
    assert allowed == ("calculator",)
    with pytest.raises(ValueError, match="allowed_tools"):
        resolve_allowed_tool_names(runtime_specs=specs, allowed_tools=["bad"])

    text = build_tool_choices_text(choices=choices)
    assert "tool_name" in text

    cloned = clone_tool_choice(choices[0])
    assert cloned == choices[0]
    assert cloned is not choices[0]

    prompt = build_tool_call_prompt(
        prompt="task",
        choices_block="- tool_name: calculator",
        prompt_template="choices: $choices_block\nquestion: $user_prompt",
    )
    assert "question: task" in prompt

    parsed_from_response = parse_tool_call_from_response(
        LLMResponse(
            text="",
            tool_calls=(ToolCall(name="calculator", arguments_json='{"a": 1}', call_id="c1"),),
        )
    )
    assert parsed_from_response == {
        "tool_name": "calculator",
        "tool_input": {"a": 1},
        "call_id": "c1",
    }

    parsed_from_bad_response = parse_tool_call_from_response(
        LLMResponse(
            text="",
            tool_calls=(ToolCall(name="calculator", arguments_json="{bad-json", call_id="c1"),),
        )
    )
    assert parsed_from_bad_response == {
        "tool_name": "calculator",
        "tool_input": "{bad-json",
        "call_id": "c1",
    }

    assert parse_tool_call("not-json") is None
    parsed_text = parse_tool_call('{"tool_name": "calculator", "tool_input": {"a": 1}}')
    assert isinstance(parsed_text, dict)

    selected = select_tool_choice(parsed_tool_call=parsed_text, choices=choices)
    assert selected is not None
    assert selected[0].tool_name == "calculator"
    assert selected[1] == "model"
    assert select_tool_choice(parsed_tool_call={"tool_name": "missing"}, choices=choices) is None
    assert (
        select_tool_choice(
            parsed_tool_call={"action": "STOP", "tool_name": "calculator"},
            choices=choices,
        )
        is None
    )

    assert (
        select_tool_choice(
            parsed_tool_call={"action": "TOOL_CALL", "name": "calculator"},
            choices=choices,
        )
        is None
    )
    assert (
        select_tool_choice(
            parsed_tool_call={"action": "TOOL_CALL", "tool_names": [1, " text.word_count "]},
            choices=choices,
        )
        is None
    )

    resolved_from_model = resolve_tool_input(
        selected_choice=choices[0],
        parsed_tool_call={"tool_input": {"a": 1}},
        input_payload={"tool_input": {"a": 2}},
    )
    assert resolved_from_model == {"a": 1}

    resolved_from_input = resolve_tool_input(
        selected_choice=choices[0],
        parsed_tool_call={"tool_input": "not-json"},
        input_payload={"tool_input": '{"a": 3}'},
    )
    assert resolved_from_input == {"a": 3}

    resolved_known = resolve_tool_input(
        selected_choice=ToolChoice(tool_name="calculator", description="", input_schema={}),
        parsed_tool_call=None,
        input_payload={"prompt": "1+2"},
    )
    assert resolved_known == {"expression": "1+2"}

    resolved_from_schema_mismatch = resolve_tool_input(
        selected_choice=ToolChoice(
            tool_name="fs.read_text",
            description="",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        parsed_tool_call={"tool_input": {"text": "wrong field"}},
        input_payload={"prompt": "Read README.md and summarize it."},
    )
    assert resolved_from_schema_mismatch == {"path": "README.md"}

    assert coerce_tool_input("bad") is None

    schema = tool_call_response_schema(["calculator", "text.word_count"])
    assert "enum" in str(schema)

    class _LLMClient:
        def default_model(self) -> str:
            return "m"

        def close(self) -> None:
            return None

        def __enter__(self) -> _LLMClient:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            self.close()
            return None

        def generate(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(text=f"ok:{request.model}")

    response = request_tool_call_response(
        llm_client=_LLMClient(),
        llm_request=LLMRequest(messages=[LLMMessage(role="user", content="x")], model="m"),
    )
    assert response.text == "ok:m"
