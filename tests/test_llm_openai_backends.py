from __future__ import annotations

import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    LLMBadResponseError,
    LLMInvalidRequestError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolSpec
from design_research_agents.llm.backends.providers import openai_compatible_http as oai_http
from design_research_agents.llm.backends.providers import openai_service as oai_service
from design_research_agents.llm.structured_output import StructuredOutputResult


class _ResponseContext:
    def __init__(self, *, body: str = "{}", lines: list[bytes] | None = None) -> None:
        self._body = body.encode("utf-8")
        self._lines = lines or []

    def __enter__(self) -> _ResponseContext:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        del exc_type, exc, tb
        return False

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(self._lines)


class _DumpObj:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)

    def model_dump(self) -> dict[str, object]:
        return dict(self.__dict__)


class _CompletionsStub:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _caps(
    *, streaming: bool = True, tool_calling: str = "native", json_mode: str = "native"
) -> BackendCapabilities:
    return BackendCapabilities(
        streaming=streaming,
        tool_calling=tool_calling,  # type: ignore[arg-type]
        json_mode=json_mode,  # type: ignore[arg-type]
        vision=False,
        max_context_tokens=None,
    )


def _request(**overrides: object) -> LLMRequest:
    payload = {
        "messages": [LLMMessage(role="user", content="hello")],
        "model": "gpt-test",
        "temperature": None,
        "max_tokens": None,
        "tools": (),
        "response_schema": None,
        "response_format": None,
        "metadata": {},
        "provider_options": {},
        "task_profile": None,
    }
    payload.update(overrides)
    return LLMRequest(**payload)


def _tool(name: str = "calculator") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Compute arithmetic.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


def test_openai_http_backend_chat_url_headers_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = oai_http.OpenAICompatibleHTTPBackend(
        name="compat",
        base_url="https://host/api",
        default_model="m",
        api_key_env="COMPAT_KEY",
        api_key=None,
        capabilities=_caps(),
        config_hash="cfg",
    )
    assert backend._chat_url == "https://host/api/v1/chat/completions"

    monkeypatch.setenv("COMPAT_KEY", "env-secret")
    assert backend._headers()["Authorization"] == "Bearer env-secret"

    request = _request(
        temperature=0.2,
        max_tokens=42,
        tools=(_tool(),),
        response_schema={"type": "object"},
        provider_options={"extra": True},
    )
    payload = backend._build_payload(request, include_response_format=True)

    assert payload["model"] == "gpt-test"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 42
    assert payload["tools"][0]["function"]["name"] == "calculator"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["extra"] is True


def test_openai_http_backend_generate_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = oai_http.OpenAICompatibleHTTPBackend(
        name="compat",
        base_url="https://api.example/v1",
        default_model="m",
        api_key_env="COMPAT_KEY",
        api_key="explicit",
        capabilities=_caps(),
        config_hash="cfg",
    )

    completion_payload = {
        "choices": [{"message": {"content": "  hi  "}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 3},
    }
    monkeypatch.setattr(oai_http, "_post_json", lambda *args, **kwargs: completion_payload)

    response = backend._generate(_request())
    assert response.text == "hi"
    assert response.provider == "compat"
    assert response.usage is not None

    stream_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "calc", "arguments": '{"x":1}'}}
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 1},
    }
    stream_lines = [
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n',
        b"\n",
        f"data: {json.dumps(stream_chunk)}\n".encode(),
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]
    monkeypatch.setattr(
        oai_http,
        "_post_stream",
        lambda *args, **kwargs: _ResponseContext(lines=stream_lines),
    )

    deltas = list(backend._stream(_request()))
    assert deltas[0].text_delta == "he"
    assert deltas[1].tool_call_delta is not None
    assert deltas[1].tool_call_delta.call_id == "c1"
    assert deltas[2].usage_delta is not None
    assert deltas[2].usage_delta.prompt_tokens == 1


def test_openai_http_post_helpers_and_response_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _ok_urlopen(request: object, timeout: float) -> _ResponseContext:
        captured["timeout"] = timeout
        captured["url"] = getattr(request, "full_url", "")
        captured["method"] = getattr(request, "method", "")
        captured["payload"] = json.loads(getattr(request, "data", b"{}").decode("utf-8"))
        return _ResponseContext(body='{"ok": true}')

    monkeypatch.setattr(oai_http, "urlopen", _ok_urlopen)
    parsed = oai_http._post_json("https://unit", {"a": 1}, headers={"X": "1"})
    assert parsed == {"ok": True}
    assert captured["method"] == "POST"

    monkeypatch.setattr(oai_http, "urlopen", lambda *args, **kwargs: _ResponseContext(body="[]"))
    with pytest.raises(LLMInvalidRequestError, match="JSON object"):
        oai_http._post_json("https://unit", {}, headers={})

    err = HTTPError(
        url="https://unit",
        code=400,
        msg="bad",
        hdrs=None,
        fp=io.BytesIO(b'{"error":{"message":"invalid payload"}}'),
    )
    monkeypatch.setattr(oai_http, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(err))
    with pytest.raises(LLMInvalidRequestError, match="invalid payload"):
        oai_http._post_json("https://unit", {}, headers={})

    monkeypatch.setattr(
        oai_http,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(LLMProviderError):
        oai_http._post_stream("https://unit", {}, headers={})

    events = list(
        oai_http._iter_sse_events(
            [
                b'data: {"a":1}\n',
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ]
        )
    )
    assert events == ['{"a":1}', "[DONE]"]

    response = oai_http._parse_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "  done  ",
                        "tool_calls": [
                            {
                                "id": "call-x",
                                "function": {"name": "calculator", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 7},
        },
        _request(),
        provider="compat",
    )
    assert response.text == "done"
    assert response.tool_calls[0].name == "calculator"

    with pytest.raises(LLMInvalidRequestError, match="no choices"):
        oai_http._parse_completion_response({}, _request(), provider="compat")

    message_payloads = oai_http._format_messages(
        [
            LLMMessage(role="user", content="hello", name="alice", tool_call_id="tc1"),
            object(),
        ]
    )
    assert message_payloads == [
        {"role": "user", "content": "hello", "name": "alice", "tool_call_id": "tc1"}
    ]

    assert oai_http._format_response_format(_request(response_format={"type": "json_object"})) == {
        "type": "json_object"
    }
    assert oai_http._format_response_format(_request(response_schema={"type": "object"})) == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": {"type": "object"}},
    }
    assert oai_http._format_response_format(_request()) is None

    assert (
        oai_http._extract_tool_call_deltas([{"id": "c1", "function": {"name": "x"}}])[0].name == "x"
    )
    assert oai_http._extract_tool_call_deltas("bad") == []


def test_openai_service_backend_retry_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = oai_service.OpenAIServiceBackend(
        name="openai",
        default_model="gpt-test",
        api_key_env="OPENAI_API_KEY",
        api_key="local-key",
        base_url=None,
        config_hash="cfg",
        max_retries=2,
    )

    completions = _CompletionsStub([LLMRateLimitError("rate limit"), "ok"])
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sleeps: list[float] = []
    monkeypatch.setattr(oai_service.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = backend._call_with_retry({"model": "gpt-test"})
    assert result == "ok"
    assert sleeps == [0.5]

    failing = _CompletionsStub([ValueError("bad payload")])
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=failing))
    with pytest.raises(LLMInvalidRequestError):
        backend._call_with_retry({"model": "gpt-test"})

    expected = LLMResponse(text="fallback", model="gpt-test", provider="openai")
    monkeypatch.setattr(
        backend,
        "_call_with_retry",
        lambda _payload: (_ for _ in ()).throw(
            LLMInvalidRequestError("response_format unsupported")
        ),
    )
    monkeypatch.setattr(backend, "_fallback_prompt_validate", lambda _request: expected)

    fallback_response = backend._generate(_request(response_schema={"type": "object"}))
    assert fallback_response.text == "fallback"


def test_openai_service_backend_stream_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = oai_service.OpenAIServiceBackend(
        name="openai",
        default_model="gpt-test",
        api_key_env="OPENAI_API_KEY",
        api_key=None,
        base_url="https://example",
        config_hash="cfg",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert backend._resolve_api_key() == "env-key"

    request = _request(
        temperature=0.4,
        max_tokens=32,
        tools=(_tool(),),
        response_schema={"type": "object"},
        provider_options={"seed": 123},
    )
    payload = backend._build_payload(request, include_response_format=True)
    assert payload["tools"][0]["function"]["name"] == "calculator"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["seed"] == 123

    stream = [
        _DumpObj(choices=[]),
        _DumpObj(
            choices=[
                _DumpObj(
                    delta=_DumpObj(
                        content="he",
                        tool_calls=[
                            _DumpObj(
                                id="call1",
                                function=_DumpObj(name="calculator", arguments='{"x":1}'),
                            )
                        ],
                    )
                )
            ],
            usage={"total_tokens": 2},
        ),
    ]
    monkeypatch.setattr(backend, "_call_with_retry", lambda _payload: stream)

    deltas = list(backend._stream(_request()))
    assert deltas[0].text_delta == "he"
    assert deltas[1].tool_call_delta is not None
    assert deltas[1].tool_call_delta.name == "calculator"
    assert deltas[2].usage_delta is not None

    completion = _DumpObj(
        choices=[
            _DumpObj(
                message=_DumpObj(
                    content=" done ",
                    tool_calls=[
                        {"id": "c1", "function": {"name": "calculator", "arguments": "{}"}}
                    ],
                ),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    parsed = oai_service._parse_completion_response(completion, _request(), provider="openai")
    assert parsed.text == "done"
    assert parsed.tool_calls[0].call_id == "c1"
    assert parsed.usage is not None

    with pytest.raises(LLMInvalidRequestError, match="no choices"):
        oai_service._parse_completion_response(_DumpObj(choices=[]), _request(), provider="openai")

    assert oai_service._tool_calls_to_list(None) is None
    assert oai_service._tool_calls_to_list([{"id": "x"}]) == [{"id": "x"}]
    assert oai_service._tool_calls_to_list([_DumpObj(id="x")]) == [{"id": "x"}]
    assert oai_service._usage_to_dict({"total_tokens": 1}) == {"total_tokens": 1}
    assert oai_service._usage_to_dict(_DumpObj(total_tokens=3)) == {"total_tokens": 3}
    assert oai_service._usage_to_dict(object()) is None

    assert oai_service._is_response_format_error(ValueError("bad response_format")) is True
    assert oai_service._is_response_format_error(ValueError("other")) is False
    assert oai_service._should_retry(LLMRateLimitError("x")) is True
    assert oai_service._should_retry(LLMProviderError("x")) is True
    assert oai_service._should_retry(LLMInvalidRequestError("x")) is False


def test_openai_service_merge_structured_response_and_generate_json_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = LLMResponse(text="", model="m", provider="openai", raw={})
    structured = StructuredOutputResult(response=base, parsed={"answer": 1}, attempts=1)

    merged = oai_service._merge_structured_response(structured)
    assert merged.text == '{"answer": 1}'
    assert merged.raw is not None
    assert merged.raw["structured_output"]["attempts"] == 2

    backend = oai_service.OpenAIServiceBackend(
        name="openai",
        default_model="m",
        api_key_env="OPENAI_API_KEY",
        api_key="k",
        base_url=None,
        config_hash="cfg",
    )
    monkeypatch.setattr(
        backend,
        "_generate_without_response_format",
        lambda request: LLMResponse(text="{}", model=request.model, provider="openai"),
    )
    monkeypatch.setattr(
        oai_service,
        "generate_json",
        lambda **kwargs: StructuredOutputResult(
            response=LLMResponse(text="", model="m", provider="openai", raw={}),
            parsed={"ok": True},
            attempts=0,
        ),
    )

    fallback = backend._fallback_prompt_validate(_request(response_schema={"type": "object"}))
    assert fallback.text == '{"ok": true}'


def test_openai_service_create_client_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = oai_service.OpenAIServiceBackend(
        name="openai",
        default_model="m",
        api_key_env="OPENAI_API_KEY",
        api_key=None,
        base_url=None,
        config_hash="cfg",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        backend._resolve_api_key()

    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "openai":
            raise ImportError("missing openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="openai"):
        backend._create_client()


def test_openai_service_response_to_dict_handles_fallback() -> None:
    class _NoDump:
        def __str__(self) -> str:
            return "raw-string"

    class _BadDump:
        def model_dump(self) -> object:
            raise LLMBadResponseError("boom")

    assert oai_service._response_to_dict(_NoDump()) == {"raw": "raw-string"}
    fallback = oai_service._response_to_dict(_BadDump())
    assert isinstance(fallback["raw"], str)
    assert "_BadDump object" in fallback["raw"]
