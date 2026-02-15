from __future__ import annotations

import builtins

import pytest

from design_research_agents.contracts.llm import (
    LLMAuthError,
    LLMBadResponseError,
    LLMInvalidRequestError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolSpec
from design_research_agents.llm.backends import utils as backend_utils
from design_research_agents.llm.backends.errors import map_backend_exception
from design_research_agents.llm.structured_output import (
    StructuredOutputResult,
    _build_json_instruction,
    _parse_json_strict,
    _validate_json,
    _with_instruction,
    build_tool_call_instruction,
    generate_json,
)


def _request() -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role="user", content="hello")],
        model="m",
        temperature=0.1,
        max_tokens=50,
        tools=(),
        response_schema=None,
        response_format={"type": "json_object"},
        metadata={"trace_id": "x"},
        provider_options={"seed": 1},
        task_profile=None,
    )


def test_backend_utils_messages_usage_and_tool_calls() -> None:
    prompt = backend_utils.messages_to_prompt(
        [
            LLMMessage(role="system", content="rules"),
            LLMMessage(role="user", content="hello"),
        ]
    )
    assert prompt == "system: rules\nuser: hello"

    assert backend_utils.parse_usage(None) is None
    assert (
        backend_utils.parse_usage(
            {"prompt_tokens": 2.0, "completion_tokens": 1, "total_tokens": True}
        )
        is not None
    )
    usage = backend_utils.parse_usage(
        {"prompt_tokens": 2.0, "completion_tokens": 1, "total_tokens": True}
    )
    assert usage is not None
    assert usage.prompt_tokens == 2
    assert usage.completion_tokens == 1
    assert usage.total_tokens is None
    assert backend_utils.parse_usage({"prompt_tokens": "bad"}) is None

    calls = backend_utils.parse_tool_calls(
        [
            {"function": {"name": "calc", "arguments": "{}"}},
            {"id": "call-2", "function": {"name": "search", "arguments": '{"q":"x"}'}},
            {"function": {"name": "", "arguments": "{}"}},
            "bad",
        ]
    )
    assert len(calls) == 2
    assert calls[0].call_id == "call_1"
    assert calls[1].call_id == "call-2"
    assert backend_utils.parse_tool_calls("bad") == ()


def test_map_backend_exception_variants() -> None:
    class _StatusError(Exception):
        def __init__(self, status_code: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code

    assert isinstance(map_backend_exception(LLMInvalidRequestError("x")), LLMInvalidRequestError)
    assert isinstance(map_backend_exception(_StatusError(401, "unauthorized")), LLMAuthError)
    assert isinstance(map_backend_exception(_StatusError(429, "rate")), LLMRateLimitError)
    assert isinstance(map_backend_exception(_StatusError(404, "missing")), LLMInvalidRequestError)
    assert isinstance(map_backend_exception(Exception("API key not set")), LLMAuthError)
    assert isinstance(map_backend_exception(Exception("hit RATE LIMIT")), LLMRateLimitError)
    assert isinstance(map_backend_exception(ValueError("bad input")), LLMInvalidRequestError)
    assert isinstance(map_backend_exception(Exception("other")), LLMProviderError)


def test_generate_json_success_retry_and_failure() -> None:
    attempts: list[str] = []

    def _generate(request: LLMRequest) -> LLMResponse:
        attempts.append(request.messages[-1].content)
        if len(attempts) == 1:
            return LLMResponse(text="not json", model="m", provider="p")
        return LLMResponse(text='{"ok": true}', model="m", provider="p")

    result = generate_json(
        generate_fn=_generate,
        request=_request(),
        schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
        max_retries=1,
    )
    assert isinstance(result, StructuredOutputResult)
    assert result.parsed == {"ok": True}
    assert result.attempts == 1
    assert "previous response was invalid" in attempts[1].lower()

    with pytest.raises(LLMBadResponseError, match="Failed to produce valid JSON"):
        generate_json(
            generate_fn=lambda _request: LLMResponse(text="", model="m", provider="p"),
            request=_request(),
            schema=None,
            max_retries=0,
        )

    with pytest.raises(ValueError, match=">= 0"):
        generate_json(
            generate_fn=lambda _request: LLMResponse(text="{}", model="m", provider="p"),
            request=_request(),
            schema=None,
            max_retries=-1,
        )


def test_structured_output_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_instruction = build_tool_call_instruction(
        (
            ToolSpec(
                name="calculator",
                description="compute",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            ),
        )
    )
    assert "Available tools" in tool_instruction
    assert "calculator" in tool_instruction

    instruction = _build_json_instruction(
        schema={"type": "object"},
        extra_instructions="Only output data",
        error_message="bad json",
    )
    assert "draft 2020-12" in instruction
    assert "Only output data" in instruction
    assert "bad json" in instruction

    adjusted = _with_instruction(_request(), "Return JSON")
    assert adjusted.messages[-1].role == "system"
    assert adjusted.response_format is None
    assert adjusted.tools == ()
    assert adjusted.provider_options == {"seed": 1}

    assert _parse_json_strict('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError, match="Empty response"):
        _parse_json_strict("  ")

    _validate_json(None, {"ok": True})
    _validate_json({"type": "object", "required": ["a"]}, {"a": 1})
    with pytest.raises(ValueError, match="at"):
        _validate_json(
            {
                "type": "object",
                "required": ["a"],
                "properties": {"a": {"type": "integer"}},
            },
            {"a": "bad"},
        )

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "jsonschema":
            raise ImportError("missing jsonschema")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="jsonschema"):
        _validate_json({"type": "object"}, {})
