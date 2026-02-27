from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from design_research_agents._contracts._llm import (
    LLMBadResponseError,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
)
from design_research_agents.llm._backends._providers import _anthropic_service as anthropic_service
from design_research_agents.llm._structured_output import StructuredOutputResult
from tests._llm_openai_backends_test_helpers import DumpObj, request, tool


class _MessagesStub:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test__anthropic_service_backend_retry_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = anthropic_service.AnthropicServiceBackend(
        name="anthropic",
        default_model="claude-3-5-haiku-latest",
        api_key_env="ANTHROPIC_API_KEY",
        api_key="local-key",
        base_url=None,
        config_hash="cfg",
        max_retries=2,
    )

    messages = _MessagesStub(
        [
            LLMRateLimitError("rate limit"),
            DumpObj(content=[DumpObj(type="text", text="ok")], stop_reason="end_turn"),
        ]
    )
    backend._client = SimpleNamespace(messages=messages)
    sleeps: list[float] = []
    monkeypatch.setattr(anthropic_service.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = backend._call_with_retry({"model": "claude-3-5-haiku-latest", "messages": []})
    assert isinstance(result, DumpObj)
    assert sleeps == [0.5]

    failing = _MessagesStub([ValueError("bad payload")])
    backend._client = SimpleNamespace(messages=failing)
    with pytest.raises(LLMInvalidRequestError):
        backend._call_with_retry({"model": "claude-3-5-haiku-latest", "messages": []})

    expected = LLMResponse(text="fallback", model="claude-3-5-haiku-latest", provider="anthropic")
    monkeypatch.setattr(
        backend,
        "_call_with_retry",
        lambda _payload: (_ for _ in ()).throw(LLMInvalidRequestError("response_format unsupported")),
    )
    monkeypatch.setattr(backend, "_fallback_prompt_validate", lambda _request: expected)

    fallback_response = backend._generate(request(response_schema={"type": "object"}))
    assert fallback_response.text == "fallback"


def test__anthropic_service_backend_stream_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = anthropic_service.AnthropicServiceBackend(
        name="anthropic",
        default_model="claude-3-5-haiku-latest",
        api_key_env="ANTHROPIC_API_KEY",
        api_key=None,
        base_url="https://api.anthropic.com",
        config_hash="cfg",
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert backend._resolve_api_key() == "env-key"

    payload = backend._build_payload(
        request(
            messages=[
                DumpObj(role="system", content="system prompt"),
                DumpObj(role="user", content="user prompt"),
                DumpObj(role="assistant", content="assistant prompt"),
                DumpObj(role="tool", content='{"ok":true}', tool_call_id="call-1", tool_name="calculator"),
            ],
            temperature=0.4,
            max_tokens=32,
            tools=(tool(),),
            response_schema={"type": "object"},
            provider_options={"metadata": {"env": "test"}},
        ),
        include_response_format=True,
    )
    assert payload["model"] == "gpt-test"
    assert payload["temperature"] == 0.4
    assert payload["max_tokens"] == 32
    assert payload["system"] == "system prompt"
    assert payload["tools"][0]["name"] == "calculator"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][2]["role"] == "user"
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"
    assert payload["metadata"] == {"env": "test"}

    stream = [
        DumpObj(type="message_start", message=DumpObj(usage=DumpObj(input_tokens=3))),
        DumpObj(type="content_block_start", index=1, content_block=DumpObj(type="tool_use", id="tu1", name="calc")),
        DumpObj(type="content_block_delta", index=0, delta=DumpObj(type="text_delta", text="he")),
        DumpObj(
            type="content_block_delta",
            index=1,
            delta=DumpObj(type="input_json_delta", partial_json='{"x":1}'),
        ),
        DumpObj(type="message_delta", usage=DumpObj(output_tokens=4)),
    ]
    monkeypatch.setattr(backend, "_call_with_retry", lambda _payload: stream)

    deltas = list(backend._stream(request()))
    assert any(delta.text_delta == "he" for delta in deltas)
    tool_delta = next(delta.tool_call_delta for delta in deltas if delta.tool_call_delta is not None)
    assert tool_delta.call_id == "tu1"
    assert tool_delta.name == "calc"
    assert tool_delta.arguments_json_delta == '{"x":1}'
    assert any(delta.usage_delta and delta.usage_delta.prompt_tokens == 3 for delta in deltas)
    assert any(delta.usage_delta and delta.usage_delta.completion_tokens == 4 for delta in deltas)

    completion = DumpObj(
        content=[
            DumpObj(type="text", text=" done "),
            DumpObj(type="tool_use", id="c1", name="calculator", input={"value": 1}),
        ],
        usage=DumpObj(input_tokens=1, output_tokens=2),
        stop_reason="end_turn",
        model="claude-3-5-haiku-latest",
    )
    parsed = anthropic_service._parse_message_response(completion, request(), provider="anthropic")
    assert parsed.text == "done"
    assert parsed.finish_reason == "end_turn"
    assert parsed.tool_calls[0].call_id == "c1"
    assert parsed.tool_calls[0].arguments_json == '{"value": 1}'
    assert parsed.usage is not None

    assert anthropic_service._usage_to_dict({"input_tokens": 1, "output_tokens": 2}) == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert anthropic_service._usage_to_dict(DumpObj(input_tokens=1, output_tokens=2)) == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert anthropic_service._usage_to_dict(object()) is None
    assert anthropic_service._is_response_format_error(ValueError("bad response_format")) is True
    assert anthropic_service._is_response_format_error(ValueError("other")) is False
    assert anthropic_service._should_retry(LLMRateLimitError("x")) is True
    assert anthropic_service._should_retry(LLMProviderError("x")) is True
    assert anthropic_service._should_retry(LLMInvalidRequestError("x")) is False


def test__anthropic_service_merge_structured_response_and_generate_json_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = LLMResponse(text="", model="m", provider="anthropic", raw={})
    structured = StructuredOutputResult(response=base, parsed={"answer": 1}, attempts=1)

    merged = anthropic_service._merge_structured_response(structured)
    assert merged.text == '{"answer": 1}'
    assert merged.raw is not None
    assert merged.raw["structured_output"]["attempts"] == 2

    backend = anthropic_service.AnthropicServiceBackend(
        name="anthropic",
        default_model="m",
        api_key_env="ANTHROPIC_API_KEY",
        api_key="k",
        base_url=None,
        config_hash="cfg",
    )
    monkeypatch.setattr(
        backend,
        "_generate_without_response_format",
        lambda request_obj: LLMResponse(text="{}", model=request_obj.model, provider="anthropic"),
    )
    monkeypatch.setattr(
        anthropic_service,
        "generate_json",
        lambda **kwargs: StructuredOutputResult(
            response=LLMResponse(text="", model="m", provider="anthropic", raw={}),
            parsed={"ok": True},
            attempts=0,
        ),
    )

    fallback = backend._fallback_prompt_validate(request(response_schema={"type": "object"}))
    assert fallback.text == '{"ok": true}'


def test__anthropic_service_create_client_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = anthropic_service.AnthropicServiceBackend(
        name="anthropic",
        default_model="m",
        api_key_env="ANTHROPIC_API_KEY",
        api_key=None,
        base_url=None,
        config_hash="cfg",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        backend._resolve_api_key()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "anthropic":
            raise ImportError("missing anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="anthropic"):
        backend._create_client()


def test__anthropic_service_response_to_dict_handles_fallback() -> None:
    class _NoDump:
        def __str__(self) -> str:
            return "raw-string"

    class _BadDump:
        def model_dump(self) -> object:
            raise LLMBadResponseError("boom")

    assert anthropic_service._response_to_dict(_NoDump()) == {"raw": "raw-string"}
    fallback = anthropic_service._response_to_dict(_BadDump())
    assert isinstance(fallback["raw"], str)
    assert "_BadDump object" in fallback["raw"]


def test__anthropic_service_create_client_passes_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = anthropic_service.AnthropicServiceBackend(
        name="anthropic",
        default_model="m",
        api_key_env="ANTHROPIC_API_KEY",
        api_key="key",
        base_url="https://example.anthropic.local",
        config_hash="cfg",
    )
    captured: dict[str, object] = {}

    class _AnthropicStub:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "anthropic", SimpleNamespace(Anthropic=_AnthropicStub))
    _ = backend._create_client()
    assert captured["api_key"] == "key"
    assert captured["base_url"] == "https://example.anthropic.local"
