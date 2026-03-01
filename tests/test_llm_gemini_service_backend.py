from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from design_research_agents._contracts._llm import (
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
)
from design_research_agents.llm._backends._providers import _gemini_service as gemini_service
from design_research_agents.llm._structured_output import StructuredOutputResult
from tests._llm_openai_backends_test_helpers import DumpObj, request


class _ModelsStub:
    def __init__(self, *, generate_outcomes: list[object], stream_outcomes: list[object] | None = None) -> None:
        self._generate_outcomes = list(generate_outcomes)
        self._stream_outcomes = list(stream_outcomes or [])
        self.generate_calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.generate_calls.append(dict(kwargs))
        outcome = self._generate_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def generate_content_stream(self, **kwargs: object) -> object:
        self.stream_calls.append(dict(kwargs))
        if self._stream_outcomes:
            outcome = self._stream_outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return []


def test__gemini_service_backend_retry_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = gemini_service.GeminiServiceBackend(
        name="gemini",
        default_model="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY",
        api_key="local-key",
        config_hash="cfg",
        max_retries=2,
    )

    models = _ModelsStub(generate_outcomes=[LLMRateLimitError("rate limit"), DumpObj(text="ok", candidates=[])])
    backend._client = SimpleNamespace(models=models)
    sleeps: list[float] = []
    monkeypatch.setattr(gemini_service.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = backend._call_with_retry({"model": "gemini-2.5-flash", "contents": []})
    assert isinstance(result, DumpObj)
    assert sleeps == [0.5]

    failing = _ModelsStub(generate_outcomes=[ValueError("bad payload")])
    backend._client = SimpleNamespace(models=failing)
    with pytest.raises(LLMInvalidRequestError):
        backend._call_with_retry({"model": "gemini-2.5-flash", "contents": []})

    expected = LLMResponse(text="fallback", model="gemini-2.5-flash", provider="gemini")
    monkeypatch.setattr(
        backend,
        "_call_with_retry",
        lambda _payload: (_ for _ in ()).throw(LLMInvalidRequestError("response_schema unsupported")),
    )
    monkeypatch.setattr(backend, "_fallback_prompt_validate", lambda _request: expected)

    fallback_response = backend._generate(request(response_schema={"type": "object"}))
    assert fallback_response.text == "fallback"


def test__gemini_service_backend_stream_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = gemini_service.GeminiServiceBackend(
        name="gemini",
        default_model="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY",
        api_key=None,
        config_hash="cfg",
    )

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fallback-key")
    assert backend._resolve_api_key() == "fallback-key"

    payload = backend._build_call_args(
        request(
            messages=[
                DumpObj(role="system", content="system prompt"),
                DumpObj(role="user", content="user prompt"),
                DumpObj(role="assistant", content="assistant prompt"),
            ],
            temperature=0.4,
            max_tokens=32,
            response_schema={"type": "object"},
            provider_options={"seed": 123},
        ),
        include_response_format=True,
    )
    assert payload["model"] == "gpt-test"
    assert payload["config"]["temperature"] == 0.4
    assert payload["config"]["max_output_tokens"] == 32
    assert payload["config"]["response_mime_type"] == "application/json"
    assert payload["config"]["seed"] == 123
    assert payload["config"]["system_instruction"] == "system prompt"
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][1]["role"] == "model"

    stream = [
        DumpObj(text="he", usage_metadata=None),
        DumpObj(text="", usage_metadata=DumpObj(prompt_token_count=1, candidates_token_count=2, total_token_count=3)),
    ]
    monkeypatch.setattr(backend, "_call_stream_with_retry", lambda _payload: stream)

    deltas = list(backend._stream(request()))
    assert deltas[0].text_delta == "he"
    assert deltas[1].usage_delta is not None
    assert deltas[1].usage_delta.prompt_tokens == 1

    completion = DumpObj(
        text=" done ",
        candidates=[DumpObj(finish_reason=DumpObj(value="STOP"))],
        usage_metadata=DumpObj(prompt_token_count=1, candidates_token_count=2, total_token_count=3),
    )
    parsed = gemini_service._parse_generate_content_response(completion, request(), provider="gemini")
    assert parsed.text == "done"
    assert parsed.finish_reason == "STOP"
    assert parsed.usage is not None

    assert gemini_service._usage_metadata_to_dict(None) is None
    assert gemini_service._usage_metadata_to_dict({"prompt_token_count": 1}) == {
        "prompt_tokens": 1,
        "completion_tokens": None,
        "total_tokens": None,
    }
    assert gemini_service._format_response_config(request(response_format={"type": "json_object"})) == {
        "response_mime_type": "application/json"
    }
    assert gemini_service._format_response_config(
        request(
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            }
        )
    ) == {
        "response_mime_type": "application/json",
        "response_schema": {"type": "object"},
    }

    exc = Exception("x")
    normalized = gemini_service._normalize_gemini_exception(exc)
    assert normalized is exc
    assert gemini_service._is_response_format_error(ValueError("bad response_schema")) is True
    assert gemini_service._is_response_format_error(ValueError("other")) is False
    assert gemini_service._should_retry(LLMRateLimitError("x")) is True
    assert gemini_service._should_retry(LLMProviderError("x")) is True
    assert gemini_service._should_retry(LLMInvalidRequestError("x")) is False


def test__gemini_service_merge_structured_response_and_generate_json_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = LLMResponse(text="", model="m", provider="gemini", raw={})
    structured = StructuredOutputResult(response=base, parsed={"answer": 1}, attempts=1)

    merged = gemini_service._merge_structured_response(structured)
    assert merged.text == '{"answer": 1}'
    assert merged.raw is not None
    assert merged.raw["structured_output"]["attempts"] == 2

    backend = gemini_service.GeminiServiceBackend(
        name="gemini",
        default_model="m",
        api_key_env="GOOGLE_API_KEY",
        api_key="k",
        config_hash="cfg",
    )
    monkeypatch.setattr(
        backend,
        "_generate_without_response_format",
        lambda request_obj: LLMResponse(text="{}", model=request_obj.model, provider="gemini"),
    )
    monkeypatch.setattr(
        gemini_service,
        "generate_json",
        lambda **kwargs: StructuredOutputResult(
            response=LLMResponse(text="", model="m", provider="gemini", raw={}),
            parsed={"ok": True},
            attempts=0,
        ),
    )

    fallback = backend._fallback_prompt_validate(request(response_schema={"type": "object"}))
    assert fallback.text == '{"ok": true}'


def test__gemini_service_create_client_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = gemini_service.GeminiServiceBackend(
        name="gemini",
        default_model="m",
        api_key_env="GOOGLE_API_KEY",
        api_key=None,
        config_hash="cfg",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY is not set"):
        backend._resolve_api_key()

    monkeypatch.setenv("GOOGLE_API_KEY", "secret")

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "google":
            raise ImportError("missing google")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="google-genai"):
        backend._create_client()
