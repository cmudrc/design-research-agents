from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import design_research_agents.llm.clients._managed_port_reservations as managed_port_module
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
    AnthropicServiceLLMClient,
    AzureOpenAIServiceLLMClient,
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
    with LlamaCppServerLLMClient() as llama:
        clients = (
            llama,
            AnthropicServiceLLMClient(),
            AzureOpenAIServiceLLMClient(),
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
        for client in clients:
            assert isinstance(client.default_model(), str)
            assert client.default_model().strip()


def test_provider_clients_use_expected_default_backend_names() -> None:
    with LlamaCppServerLLMClient() as llama:
        assert llama._backend.name == "llama-local"

    assert AnthropicServiceLLMClient()._backend.name == "anthropic"
    assert AzureOpenAIServiceLLMClient()._backend.name == "azure-openai"
    assert GeminiServiceLLMClient()._backend.name == "gemini"
    assert GroqServiceLLMClient()._backend.name == "groq"
    assert OpenAIServiceLLMClient()._backend.name == "openai"
    assert OpenAICompatibleHTTPLLMClient()._backend.name == "openai-compatible"
    assert TransformersLocalLLMClient()._backend.name == "transformers-local"
    assert MLXLocalLLMClient()._backend.name == "mlx-local"
    assert VLLMServerLLMClient(manage_server=False)._backend.name == "vllm-local"
    assert OllamaLLMClient(manage_server=False)._backend.name == "ollama-local"
    assert SGLangServerLLMClient(manage_server=False)._backend.name == "sglang-local"


def test_resolve_managed_server_port_falls_back_when_requested_port_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_calls: list[int] = []

    def _fake_probe(*, host: str, port: int) -> int | None:
        del host
        probe_calls.append(port)
        if port == 8001:
            return None
        return 43123

    monkeypatch.setattr(managed_port_module, "_probe_bindable_tcp_port", _fake_probe)
    assert managed_port_module._resolve_managed_server_port(host="127.0.0.1", requested_port=8001) == 43123
    assert probe_calls == [8001, 0]


def test_reserve_managed_server_port_covers_fallback_and_release_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []
    reservations = [
        None,
        managed_port_module._ReservedManagedPort(
            port=45123,
            reservation_socket=SimpleNamespace(close=lambda: closed.append("fallback")),  # type: ignore[arg-type]
        ),
        None,
        None,
    ]

    monkeypatch.setattr(
        managed_port_module,
        "_reserve_bindable_tcp_port",
        lambda *, host, port: reservations.pop(0),
    )

    fallback = managed_port_module._reserve_managed_server_port(host="127.0.0.1", requested_port=8001)
    assert fallback.port == 45123
    fallback.release()
    assert closed == ["fallback"]

    unresolved = managed_port_module._reserve_managed_server_port(host="127.0.0.1", requested_port=8001)
    assert unresolved.port == 8001
    unresolved.release()


def test_probe_bindable_tcp_port_returns_none_when_host_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        managed_port_module.socket,
        "getaddrinfo",
        lambda *args, **kwargs: (_ for _ in ()).throw(managed_port_module.socket.gaierror("bad host")),
    )
    assert managed_port_module._probe_bindable_tcp_port(host="bad-host", port=8001) is None


def test_managed_server_clients_use_auto_resolved_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shared_client_module,
        "_reserve_managed_server_port",
        lambda *, host, requested_port: managed_port_module._ReservedManagedPort(
            port=requested_port + 7,
        ),
    )

    with LlamaCppServerLLMClient(port=9100) as llama:
        assert llama._llama_server.port == 9107
        assert llama._backend.base_url == "http://127.0.0.1:9107/v1"
        assert llama.server_snapshot()["port"] == 9107

    with VLLMServerLLMClient(port=9200, manage_server=True) as vllm:
        assert vllm._vllm_server is not None
        assert vllm._vllm_server.port == 9207
        assert vllm._backend.base_url == "http://127.0.0.1:9207/v1"
        assert vllm.config_snapshot()["port"] == 9207

    with OllamaLLMClient(port=9300, manage_server=True) as ollama:
        assert ollama._ollama_server is not None
        assert ollama._ollama_server.port == 9307
        assert ollama._backend.base_url == "http://127.0.0.1:9307"
        assert ollama.config_snapshot()["port"] == 9307

    with SGLangServerLLMClient(port=9400, manage_server=True) as sglang:
        assert sglang._sglang_server is not None
        assert sglang._sglang_server.port == 9407
        assert sglang._backend.base_url == "http://127.0.0.1:9407/v1"
        assert sglang.config_snapshot()["port"] == 9407


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


def test_single_backend_client_context_manager_closes_on_exit() -> None:
    backend = _StubBackend(name="ctx-backend", default_model="ctx-model")

    class _TrackingClient(_SingleBackendLLMClient):
        def __init__(self) -> None:
            super().__init__(backend=backend)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    client = _TrackingClient()
    with client as opened:
        assert opened is client
        assert opened.closed is False
        assert opened.default_model() == "ctx-model"
        response = opened.generate(
            LLMRequest(
                messages=(LLMMessage(role="user", content="context-manager smoke"),),
                model="ctx-model",
                temperature=0.1,
                max_tokens=16,
            )
        )
        assert response.provider == "ctx-backend"

    assert client.closed is True
    request = backend.calls[-1]
    assert request.model == "ctx-model"
    assert request.temperature == 0.1
    assert request.max_tokens == 16
    assert request.response_schema is None
    assert request.provider_options == {}
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


def test_azure_provider_client_config_snapshot_contains_provider_fields() -> None:
    client = AzureOpenAIServiceLLMClient(
        name="azure-openai-prod",
        default_model="gpt-4o-mini",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_key="test-key",
        azure_endpoint_env="AZURE_OPENAI_ENDPOINT",
        azure_endpoint="https://example-resource.openai.azure.com",
        api_version_env="AZURE_OPENAI_API_VERSION",
        api_version="2025-01-01-preview",
        max_retries=5,
    )
    snapshot = client.config_snapshot()
    assert snapshot["name"] == "azure-openai-prod"
    assert snapshot["kind"] == "azure_openai_service"
    assert snapshot["base_url"] == "https://example-resource.openai.azure.com"
    assert snapshot["has_api_key"] is True
    assert snapshot["has_azure_endpoint"] is True
    assert snapshot["api_version"] == "2025-01-01-preview"


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
