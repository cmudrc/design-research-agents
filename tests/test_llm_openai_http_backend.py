from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from design_research_agents._contracts._llm import (
    LLMInvalidRequestError,
    LLMMessage,
    LLMProviderError,
)
from design_research_agents.llm._backends._providers import (
    _openai_compatible_http as oai_http,
)
from tests._llm_openai_backends_test_helpers import caps, request, tool


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


def test_openai_http_backend_chat_url_headers_and_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = oai_http.OpenAICompatibleHTTPBackend(
        name="compat",
        base_url="https://host/api",
        default_model="m",
        api_key_env="COMPAT_KEY",
        api_key=None,
        capabilities=caps(),
        config_hash="cfg",
    )
    assert backend._chat_url == "https://host/api/v1/chat/completions"
    monkeypatch.setenv("COMPAT_KEY", "env-secret")
    assert backend._headers()["Authorization"] == "Bearer env-secret"

    payload = backend._build_payload(
        request(
            temperature=0.2,
            max_tokens=42,
            tools=(tool(),),
            response_schema={"type": "object"},
            provider_options={"extra": True},
        ),
        include_response_format=True,
    )

    assert payload["model"] == "gpt-test"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 42
    assert payload["tools"][0]["function"]["name"] == "calculator"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["extra"] is True


def test_openai_http_backend_generate_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = oai_http.OpenAICompatibleHTTPBackend(
        name="compat",
        base_url="https://api.example/v1",
        default_model="m",
        api_key_env="COMPAT_KEY",
        api_key="explicit",
        capabilities=caps(),
        config_hash="cfg",
    )

    completion_payload = {
        "choices": [{"message": {"content": "  hi  "}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 3},
    }
    monkeypatch.setattr(oai_http, "_post_json", lambda *args, **kwargs: completion_payload)

    response = backend._generate(request())
    assert response.text == "hi"
    assert response.provider == "compat"
    assert response.usage is not None

    stream_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "calc", "arguments": '{"x":1}'},
                        }
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

    deltas = list(backend._stream(request()))
    assert deltas[0].text_delta == "he"
    assert deltas[1].tool_call_delta is not None
    assert deltas[1].tool_call_delta.call_id == "c1"
    assert deltas[2].usage_delta is not None
    assert deltas[2].usage_delta.prompt_tokens == 1


def test_openai_http_post_helpers_and_response_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _ok_urlopen(request_obj: object, timeout: float) -> _ResponseContext:
        captured["timeout"] = timeout
        captured["url"] = getattr(request_obj, "full_url", "")
        captured["method"] = getattr(request_obj, "method", "")
        captured["payload"] = json.loads(getattr(request_obj, "data", b"{}").decode("utf-8"))
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
        request(),
        provider="compat",
    )
    assert response.text == "done"
    assert response.tool_calls[0].name == "calculator"

    with pytest.raises(LLMInvalidRequestError, match="no choices"):
        oai_http._parse_completion_response({}, request(), provider="compat")

    message_payloads = oai_http._format_messages(
        [
            LLMMessage(role="user", content="hello", name="alice", tool_call_id="tc1"),
            object(),
        ]
    )
    assert message_payloads == [{"role": "user", "content": "hello", "name": "alice", "tool_call_id": "tc1"}]

    assert oai_http._format_response_format(request(response_format={"type": "json_object"})) == {"type": "json_object"}
    assert oai_http._format_response_format(request(response_schema={"type": "object"})) == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": {"type": "object"}},
    }
    assert oai_http._format_response_format(request()) is None

    assert oai_http._extract_tool_call_deltas([{"id": "c1", "function": {"name": "x"}}])[0].name == "x"
    assert oai_http._extract_tool_call_deltas("bad") == []
