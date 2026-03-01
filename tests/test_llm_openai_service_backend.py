from __future__ import annotations

import pytest

from design_research_agents._contracts._llm import (
    LLMBadResponseError,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
)
from design_research_agents.llm._backends._providers import _openai_service as oai_service
from design_research_agents.llm._structured_output import StructuredOutputResult
from tests._llm_openai_backends_test_helpers import (
    CompletionsStub,
    DumpObj,
    client_with_completions,
    request,
    tool,
)


def test__openai_service_backend_retry_and_fallback(monkeypatch) -> None:
    backend = oai_service.OpenAIServiceBackend(
        name="openai",
        default_model="gpt-test",
        api_key_env="OPENAI_API_KEY",
        api_key="local-key",
        base_url=None,
        config_hash="cfg",
        max_retries=2,
    )

    completions = CompletionsStub([LLMRateLimitError("rate limit"), "ok"])
    backend._client = client_with_completions(completions)
    sleeps: list[float] = []
    monkeypatch.setattr(oai_service.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = backend._call_with_retry({"model": "gpt-test"})
    assert result == "ok"
    assert sleeps == [0.5]

    failing = CompletionsStub([ValueError("bad payload")])
    backend._client = client_with_completions(failing)
    with pytest.raises(LLMInvalidRequestError):
        backend._call_with_retry({"model": "gpt-test"})

    expected = LLMResponse(text="fallback", model="gpt-test", provider="openai")
    monkeypatch.setattr(
        backend,
        "_call_with_retry",
        lambda _payload: (_ for _ in ()).throw(LLMInvalidRequestError("response_format unsupported")),
    )
    monkeypatch.setattr(backend, "_fallback_prompt_validate", lambda _request: expected)

    fallback_response = backend._generate(request(response_schema={"type": "object"}))
    assert fallback_response.text == "fallback"


def test__openai_service_backend_stream_and_helpers(monkeypatch) -> None:
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

    payload = backend._build_payload(
        request(
            temperature=0.4,
            max_tokens=32,
            tools=(tool(),),
            response_schema={"type": "object"},
            provider_options={"seed": 123},
        ),
        include_response_format=True,
    )
    assert payload["tools"][0]["function"]["name"] == "calculator"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["seed"] == 123

    stream = [
        DumpObj(choices=[]),
        DumpObj(
            choices=[
                DumpObj(
                    delta=DumpObj(
                        content="he",
                        tool_calls=[
                            DumpObj(
                                id="call1",
                                function=DumpObj(name="calculator", arguments='{"x":1}'),
                            )
                        ],
                    )
                )
            ],
            usage={"total_tokens": 2},
        ),
    ]
    monkeypatch.setattr(backend, "_call_with_retry", lambda _payload: stream)

    deltas = list(backend._stream(request()))
    assert deltas[0].text_delta == "he"
    assert deltas[1].tool_call_delta is not None
    assert deltas[1].tool_call_delta.name == "calculator"
    assert deltas[2].usage_delta is not None

    completion = DumpObj(
        choices=[
            DumpObj(
                message=DumpObj(
                    content=" done ",
                    tool_calls=[
                        {
                            "id": "c1",
                            "function": {"name": "calculator", "arguments": "{}"},
                        }
                    ],
                ),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    parsed = oai_service._parse_completion_response(completion, request(), provider="openai")
    assert parsed.text == "done"
    assert parsed.tool_calls[0].call_id == "c1"
    assert parsed.usage is not None

    with pytest.raises(LLMInvalidRequestError, match="no choices"):
        oai_service._parse_completion_response(DumpObj(choices=[]), request(), provider="openai")

    assert oai_service._tool_calls_to_list(None) is None
    assert oai_service._tool_calls_to_list([{"id": "x"}]) == [{"id": "x"}]
    assert oai_service._tool_calls_to_list([DumpObj(id="x")]) == [{"id": "x"}]
    assert oai_service._usage_to_dict({"total_tokens": 1}) == {"total_tokens": 1}
    assert oai_service._usage_to_dict(DumpObj(total_tokens=3)) == {"total_tokens": 3}
    assert oai_service._usage_to_dict(object()) is None

    assert oai_service._is_response_format_error(ValueError("bad response_format")) is True
    assert oai_service._is_response_format_error(ValueError("other")) is False
    assert oai_service._should_retry(LLMRateLimitError("x")) is True
    assert oai_service._should_retry(LLMProviderError("x")) is True
    assert oai_service._should_retry(LLMInvalidRequestError("x")) is False


def test__openai_service_merge_structured_response_and_generate_json_fallback(
    monkeypatch,
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
        lambda request_obj: LLMResponse(text="{}", model=request_obj.model, provider="openai"),
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

    fallback = backend._fallback_prompt_validate(request(response_schema={"type": "object"}))
    assert fallback.text == '{"ok": true}'


def test__openai_service_create_client_import_error(monkeypatch) -> None:
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


def test__openai_service_response_to_dict_handles_fallback() -> None:
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
