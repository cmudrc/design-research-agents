from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from design_research_agents._contracts._llm import (
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    Usage,
)
from design_research_agents.llm._backends._providers import _anthropic_service as anthropic_service
from design_research_agents.llm._backends._providers import _gemini_service as gemini_service
from design_research_agents.llm._backends._providers import _ollama_local
from design_research_agents.llm._backends._providers import _openai_service as openai_service
from tests._llm_openai_backends_test_helpers import DumpObj, request, tool


class _GeminiModels:
    def __init__(self, outcomes: list[object], stream_outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes)
        self.stream_outcomes = list(stream_outcomes or [])

    def generate_content(self, **_kwargs: object) -> object:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def generate_content_stream(self, **_kwargs: object) -> object:
        outcome = self.stream_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_gemini_backend_and_helper_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = gemini_service.GeminiServiceBackend(
        name="gemini",
        default_model="m",
        api_key_env="GOOGLE_API_KEY",
        api_key="key",
        config_hash="cfg",
        max_retries=1,
    )
    assert backend.capabilities().streaming is True
    assert backend.healthcheck().ok is True

    monkeypatch.setattr(
        backend,
        "_call_with_retry",
        lambda _payload: DumpObj(text=" generated ", candidates=[DumpObj(finish_reason="STOP")]),
    )
    assert backend._generate(request()).text == "generated"

    monkeypatch.setattr(backend, "_call_with_retry", lambda _payload: (_ for _ in ()).throw(LLMProviderError("boom")))
    with pytest.raises(LLMProviderError):
        backend._generate(request())

    backend._client = SimpleNamespace(
        models=_GeminiModels(
            outcomes=[],
            stream_outcomes=[LLMRateLimitError("rate"), [DumpObj(text="ok", usage_metadata=None)]],
        )
    )
    sleeps: list[float] = []
    monkeypatch.setattr(gemini_service.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert next(iter(backend._call_stream_with_retry({"model": "m"}))).text == "ok"
    assert sleeps == [0.5]

    monkeypatch.setattr(backend, "_call_with_retry", lambda _payload: DumpObj(text=" raw ", candidates=[]))
    assert backend._generate_without_response_format(request()).text == "raw"

    assert gemini_service._format_contents([DumpObj(role="system", content="only system")]) == [
        {"role": "user", "parts": [{"text": ""}]}
    ]
    assert gemini_service._format_contents(
        [DumpObj(role="tool", content="{}", tool_name="calc", tool_call_id="c1")]
    ) == [{"role": "user", "parts": [{"text": "tool[calc]#c1: {}"}]}]
    assert gemini_service._format_response_config(
        request(response_format={"response_mime_type": "text/plain", "response_schema": {"type": "object"}})
    ) == {"response_mime_type": "text/plain", "response_schema": {"type": "object"}}
    assert gemini_service._extract_finish_reason(DumpObj(candidates=[])) is None
    assert gemini_service._extract_finish_reason(DumpObj(candidates=[DumpObj(finish_reason=None)])) is None
    assert (
        gemini_service._extract_finish_reason(DumpObj(candidates=[DumpObj(finish_reason=DumpObj(value="MAX"))]))
        == "MAX"
    )
    assert gemini_service._usage_metadata_to_dict({"prompt_token_count": True}) is None
    assert gemini_service._response_to_dict(object())["raw"].startswith("<object")
    assert gemini_service._to_dict(DumpObj(model_dump=lambda: [])) is None

    class _GeminiToDictOnly:
        def to_dict(self) -> dict[str, bool]:
            return {"ok": True}

    assert gemini_service._to_dict(_GeminiToDictOnly()) == {"ok": True}
    assert gemini_service._coerce_int(True) is None
    assert gemini_service._coerce_int(2.9) == 2

    class _CodeError(Exception):
        code = 429

    err = _CodeError("rate")
    assert gemini_service._normalize_gemini_exception(err) is err
    assert err.status_code == 429  # type: ignore[attr-defined]


def test_anthropic_backend_and_helper_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = anthropic_service.AnthropicServiceBackend(
        name="anthropic",
        default_model="m",
        api_key_env="ANTHROPIC_API_KEY",
        api_key="key",
        base_url=None,
        config_hash="cfg",
    )
    assert backend.capabilities().tool_calling == "native"
    assert backend.healthcheck().ok is True

    monkeypatch.setattr(
        backend,
        "_call_with_retry",
        lambda _payload: DumpObj(content=[DumpObj(type="text", text=" no-format ")], model="m"),
    )
    assert backend._generate_without_response_format(request()).text == "no-format"

    assert anthropic_service._format_messages([DumpObj(role="system", content="only")]) == [
        {"role": "user", "content": ""}
    ]
    assert anthropic_service._format_messages([DumpObj(role="tool", content="{}", tool_name="calc")]) == [
        {"role": "user", "content": "tool[calc]: {}"}
    ]
    assert anthropic_service._format_response_format(request()) is None
    assert anthropic_service._extract_text_and_tool_calls("bad") == ("", ())
    text, calls = anthropic_service._extract_text_and_tool_calls(
        [
            {"type": "other"},
            {"type": "tool_use", "name": "", "input": {}},
            {"type": "tool_use", "name": "calc", "input": object()},
        ]
    )
    assert text == ""
    assert calls[0].call_id == "call_1"
    assert calls[0].arguments_json.startswith("<object")

    state: dict[int, tuple[str | None, str | None]] = {}
    assert list(anthropic_service._stream_event_deltas(DumpObj(type="unknown"), tool_state_by_index=state)) == []
    anthropic_service._capture_tool_use_start_event(DumpObj(index=True), tool_state_by_index=state)
    assert state == {}
    assert anthropic_service._text_delta_from_stream_event(DumpObj(delta=DumpObj(type="text_delta", text=""))) is None
    delta = anthropic_service._tool_call_delta_from_stream_event(
        DumpObj(index=3, delta=DumpObj(type="input_json_delta", text='{"x":1}')),
        tool_state_by_index={},
    )
    assert delta is not None
    assert delta.tool_call_delta is not None
    assert delta.tool_call_delta.arguments_json_delta == '{"x":1}'

    class _ToDictOnly:
        def to_dict(self) -> dict[str, bool]:
            return {"ok": True}

    assert anthropic_service._serialize_tool_input("raw") == "raw"
    assert anthropic_service._serialize_tool_input(None) == "{}"
    assert anthropic_service._to_dict(_ToDictOnly()) == {"ok": True}
    assert anthropic_service._optional_str(123) == "123"
    assert anthropic_service._coerce_int(True) is None
    assert anthropic_service._coerce_int(2.9) == 2


def test_openai_backend_and_helper_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = openai_service.OpenAIServiceBackend(
        name="openai",
        default_model="m",
        api_key_env="OPENAI_API_KEY",
        api_key="key",
        base_url=None,
        config_hash="cfg",
    )
    assert backend.capabilities().json_mode == "native"
    assert backend.healthcheck().ok is True

    monkeypatch.setattr(
        backend,
        "_call_with_retry",
        lambda _payload: DumpObj(choices=[DumpObj(message=DumpObj(content=" no-format "), finish_reason="stop")]),
    )
    assert backend._generate_without_response_format(request()).text == "no-format"

    chunks = [
        DumpObj(choices=[]),
        DumpObj(choices=[DumpObj(delta=None)]),
        DumpObj(choices=[DumpObj(delta=DumpObj(content="he", tool_calls=None))], usage=DumpObj(model_dump=lambda: {})),
        DumpObj(
            choices=[
                DumpObj(
                    delta=DumpObj(
                        content=None,
                        tool_calls=[DumpObj(id="c1", function=DumpObj(name="calc", arguments='{"x":1}'))],
                    )
                )
            ],
            usage=DumpObj(model_dump=lambda: {"prompt_tokens": 1}),
        ),
    ]
    monkeypatch.setattr(backend, "_call_with_retry", lambda _payload: chunks)
    deltas = list(backend._stream(request()))
    assert any(delta.text_delta == "he" for delta in deltas)
    assert any(delta.tool_call_delta and delta.tool_call_delta.name == "calc" for delta in deltas)
    assert any(delta.usage_delta and delta.usage_delta.prompt_tokens == 1 for delta in deltas)

    assert openai_service._format_messages([DumpObj(role=None, content="x"), DumpObj(role="user", content=None)]) == []
    assert openai_service._format_messages([DumpObj(role="user", content="x", name="n", tool_call_id="c")]) == [
        {"role": "user", "content": "x", "name": "n", "tool_call_id": "c"}
    ]
    assert openai_service._format_tool(tool())["function"]["name"] == "calculator"
    assert openai_service._format_response_format(request()) is None
    with pytest.raises(LLMInvalidRequestError, match="no choices"):
        openai_service._parse_completion_response(DumpObj(choices=[]), request(), provider="openai")
    assert openai_service._response_to_dict(DumpObj(model_dump=lambda: [])) == {"raw": "[]"}
    assert "raw" in openai_service._response_to_dict(DumpObj(model_dump=lambda: (_ for _ in ()).throw(ValueError("x"))))
    assert openai_service._tool_calls_to_list(None) is None
    assert openai_service._tool_calls_to_list([{"id": "c"}, DumpObj(model_dump=lambda: {"id": "d"})]) == [
        {"id": "c"},
        {"id": "d"},
    ]
    assert openai_service._tool_calls_to_list(DumpObj(model_dump=lambda: {"id": "one"})) == [{"id": "one"}]
    assert openai_service._tool_calls_to_list(object()) is None
    assert openai_service._usage_to_dict(object()) is None
    assert openai_service._usage_to_dict(DumpObj(model_dump=lambda: [])) is None
    assert openai_service._is_response_format_error(ValueError("json_schema unsupported")) is True
    assert openai_service._should_retry(LLMRateLimitError("rate")) is True
    assert openai_service._should_retry(LLMInvalidRequestError("bad")) is False


class _JsonResponse:
    def __init__(self, body: str) -> None:
        self.body = body

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        del exc_type, exc, tb
        return False

    def read(self) -> bytes:
        return self.body.encode("utf-8")


def test_ollama_local_backend_and_transport_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="base_url"):
        _ollama_local.OllamaLocalBackend(
            name="ollama",
            base_url=" ",
            default_model="m",
            request_timeout_seconds=1,
            config_hash="cfg",
        )
    with pytest.raises(ValueError, match="timeout"):
        _ollama_local.OllamaLocalBackend(
            name="ollama",
            base_url="http://unit",
            default_model="m",
            request_timeout_seconds=0,
            config_hash="cfg",
        )

    started: list[bool] = []
    backend = _ollama_local.OllamaLocalBackend(
        name="ollama",
        base_url="http://unit",
        default_model="m",
        request_timeout_seconds=1,
        config_hash="cfg",
        managed_server=SimpleNamespace(start=lambda: started.append(True)),
    )
    assert backend.capabilities().json_mode == "native"
    assert backend.healthcheck().ok is True
    backend._ensure_server_ready()
    assert started == [True]

    assert _ollama_local._translate_ollama_response_format({"type": "json_schema", "schema": {"type": "object"}}) == {
        "type": "object"
    }
    assert _ollama_local._translate_ollama_response_format(
        {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
    ) == {"type": "object"}
    assert _ollama_local._translate_ollama_response_format({"type": "other", "x": 1}) == {"type": "other", "x": 1}
    assert _ollama_local._should_fallback_to_prompt_validated_json(request(), LLMInvalidRequestError("format")) is False
    assert (
        _ollama_local._should_fallback_to_prompt_validated_json(
            request(response_schema={"type": "object"}),
            LLMInvalidRequestError("json schema unsupported"),
        )
        is True
    )

    fallback_deltas = list(
        _ollama_local._stream_prompt_validated_json_fallback(
            LLMResponse(text="{}", usage={"prompt_eval_count": 1, "eval_count": 2})
        )
    )
    assert fallback_deltas[0].text_delta == "{}"
    assert fallback_deltas[1].usage_delta == Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)

    real_post_stream = _ollama_local._post_stream
    stream_outcomes: list[object] = [LLMRateLimitError("rate"), _JsonResponse('{"ok": true}')]
    sleeps: list[float] = []

    def _flaky_stream(*_args: object, **_kwargs: object) -> object:
        outcome = stream_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(_ollama_local, "_post_stream", _flaky_stream)
    monkeypatch.setattr(_ollama_local.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert _ollama_local._post_stream_with_retry("http://unit", {}, timeout_seconds=1, max_retries=1).read()
    assert sleeps == [0.5]

    monkeypatch.setattr(_ollama_local, "urlopen", lambda *_args, **_kwargs: _JsonResponse("[]"))
    with pytest.raises(LLMInvalidRequestError, match="JSON object"):
        _ollama_local._post_json("http://unit", {}, timeout_seconds=1)
    monkeypatch.setattr(_ollama_local, "_post_stream", real_post_stream)
    monkeypatch.setattr(_ollama_local, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("down")))
    with pytest.raises(LLMProviderError):
        _ollama_local._post_stream("http://unit", {}, timeout_seconds=1)

    events = list(_ollama_local._iter_json_events([b"\n", b"data: not-json\n", b'data: {"ok": true}\n']))
    assert events == [{"ok": True}]
    parsed = _ollama_local._parse_completion_response({"message": "bad"}, request(), provider="ollama")
    assert parsed.text == ""
    assert _ollama_local._parse_ollama_usage({}) is None
    assert _ollama_local._format_messages(
        [DumpObj(role=None, content="x"), DumpObj(role="user", content="x", name="n")]
    ) == [{"role": "user", "content": "x", "name": "n"}]
    auth_error = HTTPError("http://unit", 401, "auth", None, None)
    assert _ollama_local._http_error(auth_error).__class__.__name__ == "LLMAuthError"
    bad_request = HTTPError("http://unit", 400, "bad", None, None)
    bad_request.read = lambda: json.dumps({"error": "bad payload"}).encode("utf-8")  # type: ignore[method-assign]
    assert str(_ollama_local._http_error(bad_request)) == "bad payload"
    assert _ollama_local._http_error(HTTPError("http://unit", 500, "server", None, None)).__class__.__name__ == (
        "LLMProviderError"
    )
