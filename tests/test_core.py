from __future__ import annotations

from collections.abc import Iterator

import pytest

import design_research_agents.llm.clients._shared as shared_client_module
from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm._backends._base import BaseLLMBackend
from design_research_agents.llm.clients import (
    GeminiServiceLLMClient,
    GroqServiceLLMClient,
    LlamaCppServerLLMClient,
    MLXLocalLLMClient,
    OllamaLLMClient,
    OpenAICompatibleHTTPLLMClient,
    OpenAIServiceLLMClient,
    SGLangServerLLMClient,
    TransformersLocalLLMClient,
    VLLMServerLLMClient,
)
from design_research_agents.llm.clients._shared import _SingleBackendLLMClient


class _StubBackend(BaseLLMBackend):
    def __init__(
        self,
        *,
        name: str = "stub",
        default_model: str | None = "stub-model",
        stream_chunks: tuple[str, ...] = ("stub-", "chunk"),
        json_mode: str = "none",
    ) -> None:
        super().__init__(
            name=name,
            kind="echo_test",
            default_model=default_model,
            base_url=None,
            config_hash="stub-hash",
        )
        self._stream_chunks = stream_chunks
        self._json_mode = json_mode
        self.calls: list[LLMRequest] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            tool_calling="none",
            json_mode=self._json_mode,  # type: ignore[arg-type]
            vision=False,
            max_context_tokens=None,
        )

    def healthcheck(self) -> BackendStatus:
        return BackendStatus(ok=True, message="ok")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=f"{self.name}:{request.model}",
            model=request.model,
            provider=self.name,
        )

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        self.calls.append(request)
        for chunk in self._stream_chunks:
            yield LLMDelta(text_delta=chunk)


def test_provider_clients_empty_init_and_default_model() -> None:
    llama = LlamaCppServerLLMClient()
    clients = (
        llama,
        GeminiServiceLLMClient(),
        GroqServiceLLMClient(),
        OpenAIServiceLLMClient(),
        OpenAICompatibleHTTPLLMClient(),
        TransformersLocalLLMClient(),
        MLXLocalLLMClient(),
        VLLMServerLLMClient(manage_server=False),
        OllamaLLMClient(manage_server=False),
        SGLangServerLLMClient(manage_server=False),
    )
    try:
        for client in clients:
            assert isinstance(client.default_model(), str)
            assert client.default_model().strip()
    finally:
        llama.close()


def test_provider_clients_use_expected_default_backend_names() -> None:
    llama = LlamaCppServerLLMClient()
    try:
        assert llama._backend.name == "llama-local"
    finally:
        llama.close()

    assert GeminiServiceLLMClient()._backend.name == "gemini"
    assert GroqServiceLLMClient()._backend.name == "groq"
    assert OpenAIServiceLLMClient()._backend.name == "openai"
    assert OpenAICompatibleHTTPLLMClient()._backend.name == "openai-compatible"
    assert TransformersLocalLLMClient()._backend.name == "transformers-local"
    assert MLXLocalLLMClient()._backend.name == "mlx-local"
    assert VLLMServerLLMClient(manage_server=False)._backend.name == "vllm-local"
    assert OllamaLLMClient(manage_server=False)._backend.name == "ollama-local"
    assert SGLangServerLLMClient(manage_server=False)._backend.name == "sglang-local"


def test_chat_builds_request_from_chat_params() -> None:
    backend = _StubBackend(name="chat-backend", default_model="chat-model")
    client = _SingleBackendLLMClient(backend=backend)

    response = client.chat(
        [LLMMessage(role="user", content="hello")],
        model="chat-model",
        params=LLMChatParams(
            temperature=0.2,
            max_tokens=64,
            provider_options={"seed": 7},
        ),
    )

    assert response.provider == "chat-backend"
    assert backend.calls
    request = backend.calls[-1]
    assert request.model == "chat-model"
    assert request.temperature == 0.2
    assert request.max_tokens == 64
    assert request.response_schema is None
    assert request.provider_options == {"seed": 7}
    assert request.tools == ()
    assert request.response_format is None


def test_stream_chat_emits_delta_and_completed() -> None:
    backend = _StubBackend(name="stream-backend", default_model="stream-model")
    client = _SingleBackendLLMClient(backend=backend)

    events = list(
        client.stream_chat(
            [LLMMessage(role="user", content="hello")],
            model="stream-model",
            params=LLMChatParams(),
        )
    )

    assert [event.kind for event in events] == ["delta", "delta", "completed"]
    assert events[0].delta_text == "stub-"
    assert events[1].delta_text == "chunk"
    assert events[2].response is not None
    assert events[2].response.text == "stub-chunk"


def test_default_model_raises_when_backend_default_missing() -> None:
    backend = _StubBackend(default_model=None)
    client = _SingleBackendLLMClient(backend=backend)

    with pytest.raises(ValueError, match="default_model"):
        client.default_model()


def test_single_backend_client_exposes_introspection_snapshots() -> None:
    backend = _StubBackend(name="inspect-backend", default_model="inspect-model")
    client = _SingleBackendLLMClient(
        backend=backend,
        config_snapshot={"request_timeout_seconds": 45.0},
        server_snapshot={"host": "127.0.0.1", "port": 8002},
    )

    capabilities = client.capabilities()
    assert capabilities.streaming is True

    config_snapshot = client.config_snapshot()
    assert config_snapshot["name"] == "inspect-backend"
    assert config_snapshot["kind"] == "echo_test"
    assert config_snapshot["default_model"] == "inspect-model"
    assert config_snapshot["request_timeout_seconds"] == 45.0

    server_snapshot = client.server_snapshot()
    assert server_snapshot == {"host": "127.0.0.1", "port": 8002}

    description = client.describe()
    assert description["client_class"] == "_SingleBackendLLMClient"
    assert description["default_model"] == "inspect-model"
    assert isinstance(description["backend"], dict)
    assert isinstance(description["capabilities"], dict)
    assert description["server"] == {"host": "127.0.0.1", "port": 8002}


def test_provider_client_config_snapshot_contains_provider_fields() -> None:
    client = OpenAICompatibleHTTPLLMClient(
        name="compat-test",
        base_url="http://127.0.0.1:8010/v1",
        default_model="qwen-test",
        api_key_env="OPENAI_API_KEY",
        api_key="test-key",
        max_retries=5,
    )
    snapshot = client.config_snapshot()
    assert snapshot["name"] == "compat-test"
    assert snapshot["kind"] == "openai_compatible_http"
    assert snapshot["base_url"] == "http://127.0.0.1:8010/v1"
    assert snapshot["chat_url"] == "http://127.0.0.1:8010/v1/chat/completions"
    assert snapshot["has_api_key"] is True


def test_single_backend_client_emits_model_observation_events(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _StubBackend(name="trace-backend", default_model="trace-model")
    client = _SingleBackendLLMClient(backend=backend)
    captured: dict[str, list[dict[str, object]]] = {"requests": [], "responses": []}

    monkeypatch.setattr(
        shared_client_module,
        "emit_model_request_observed",
        lambda **kwargs: captured["requests"].append(dict(kwargs)),
    )
    monkeypatch.setattr(
        shared_client_module,
        "emit_model_response_observed",
        lambda **kwargs: captured["responses"].append(dict(kwargs)),
    )

    request = LLMRequest(messages=[LLMMessage(role="user", content="hello")], model="trace-model")
    _ = client.generate(request)
    _ = list(client.stream(request))

    assert len(captured["requests"]) >= 2
    assert len(captured["responses"]) >= 2
    assert captured["requests"][0]["source"] == "_SingleBackendLLMClient.generate"
    assert captured["requests"][1]["source"] == "_SingleBackendLLMClient.stream"
