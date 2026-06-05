from __future__ import annotations

from collections.abc import Iterator

import pytest

import design_research_agents.llm.clients._managed_port_reservations as managed_port_module
import design_research_agents.llm.clients._shared as shared_client_module
from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents._contracts._tools import ToolSpec
from design_research_agents.llm._backends._base import BaseLLMBackend, _build_tool_call_schema
from design_research_agents.llm._backends._providers._openai_compatible_http import _format_response_format
from design_research_agents.llm.clients import DemoLLMClient


class _CaptureBackend(BaseLLMBackend):
    def __init__(
        self,
        *,
        default_model: str,
        response_text: str = "<think>private reasoning</think> Visible answer.",
    ) -> None:
        super().__init__(
            name="capture",
            kind="test",
            default_model=default_model,
            base_url=None,
            config_hash="capture-hash",
        )
        self.response_text = response_text
        self.calls: list[LLMRequest] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            tool_calling="best_effort",
            json_mode="native",
            vision=False,
            max_context_tokens=None,
        )

    def healthcheck(self) -> BackendStatus:
        return BackendStatus(ok=True)

    def _generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(text=self.response_text, model=request.model, provider="capture", raw={"kept": True})

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        self.calls.append(request)
        yield LLMDelta(text_delta=self.response_text)


def test_demo_client_normalizes_defaults_and_strips_thinking() -> None:
    with DemoLLMClient() as client:
        default_model = client.default_model()
        backend = _CaptureBackend(default_model=default_model)
        client._backend = backend

        response = client.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content="Name one workshop activity.")],
                provider_options={"top_k": 7, "custom": "value"},
            )
        )

    assert response.text == "Visible answer."
    assert response.raw == {"kept": True}
    assert len(backend.calls) == 1
    request = backend.calls[0]
    assert request.model == default_model
    assert request.temperature == 0.7
    assert request.max_tokens == 256
    assert request.provider_options["top_p"] == 0.8
    assert request.provider_options["top_k"] == 7
    assert request.provider_options["min_p"] == 0
    assert request.provider_options["presence_penalty"] == 1.5
    assert request.provider_options["custom"] == "value"
    assert request.messages[-1].content.endswith("/no_think")


def test_demo_client_preserves_explicit_generation_controls() -> None:
    with DemoLLMClient(thinking="auto", default_provider_options={"top_p": 0.5}) as client:
        backend = _CaptureBackend(default_model=client.default_model(), response_text="plain")
        client._backend = backend

        client.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content="hello")],
                model="qwen3-0.6b-q8-demo",
                temperature=0.2,
                max_tokens=32,
                provider_options={"top_p": 0.4},
            )
        )

    request = backend.calls[0]
    assert request.temperature == 0.2
    assert request.max_tokens == 32
    assert request.provider_options["top_p"] == 0.4
    assert request.messages[-1].content == "hello"


def test_demo_client_managed_llama_defaults_and_port_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shared_client_module,
        "_reserve_managed_server_port",
        lambda *, host, requested_port: managed_port_module._ReservedManagedPort(
            port=requested_port + 9,
        ),
    )

    with DemoLLMClient(port=8100) as client:
        assert client.default_model() == "qwen3-0.6b-q8-demo"
        assert client._llama_server.model == "Qwen3-0.6B-Q8_0.gguf"
        assert client._llama_server.hf_model_repo_id == "Qwen/Qwen3-0.6B-GGUF"
        assert client._llama_server.port == 8109
        assert client._backend.base_url == "http://127.0.0.1:8109/v1"
        assert client.config_snapshot()["thinking"] == "off"


def test_demo_client_rejects_invalid_defaults() -> None:
    with pytest.raises(ValueError, match="thinking"):
        DemoLLMClient(thinking="sometimes")
    with pytest.raises(ValueError, match="default_max_tokens"):
        DemoLLMClient(default_max_tokens=0)
    with pytest.raises(ValueError, match="default_temperature"):
        DemoLLMClient(default_temperature=-0.1)


def test_llama_cpp_response_schema_uses_json_object_schema() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    request = LLMRequest(
        messages=[LLMMessage(role="user", content="answer as JSON")],
        model="qwen3-0.6b-q8-demo",
        response_schema=schema,
    )

    assert _format_response_format(request, style="llama_cpp") == {
        "type": "json_object",
        "schema": schema,
    }


def test_tool_call_schema_constrains_allowed_tool_names() -> None:
    schema = _build_tool_call_schema(
        (
            ToolSpec(
                name="text.word_count",
                description="Count words.",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            ),
        )
    )

    tool_name_schema = schema["properties"]["tool_calls"]["items"]["properties"]["name"]  # type: ignore[index]
    assert tool_name_schema["enum"] == ["text.word_count"]  # type: ignore[index]
