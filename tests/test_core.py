from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    BackendStatus,
    LLMCapabilityError,
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    TaskProfile,
)
from design_research_agents.llm import (
    BaseLLMClient,
    LLMRouter,
    configure_router_from_yaml,
    resolve_default_model,
    set_default_router,
)
from design_research_agents.llm.backends.base import BaseLLMBackend


class _StubBackend(BaseLLMBackend):
    def __init__(
        self,
        *,
        name: str,
        kind: str,
        default_model: str,
        capabilities: BackendCapabilities,
    ) -> None:
        super().__init__(
            name=name,
            kind=kind,
            default_model=default_model,
            base_url=None,
            config_hash=f"hash-{name}",
        )
        self._caps = capabilities
        self.calls: list[LLMRequest] = []

    def capabilities(self) -> BackendCapabilities:
        return self._caps

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
        yield LLMDelta(text_delta=f"{self.name}-")
        yield LLMDelta(text_delta="stream")


@pytest.fixture(autouse=True)
def _clear_default_router() -> Iterator[None]:
    set_default_router(None)
    yield
    set_default_router(None)


def _chat_request(*, model: str) -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role="user", content="hello")],
        model=model,
        temperature=None,
        max_tokens=None,
        tools=(),
        response_schema=None,
        response_format=None,
        metadata={},
        provider_options={},
        task_profile=None,
    )


def test_base_llm_client_requires_router_or_default() -> None:
    with pytest.raises(ValueError, match="No LLM router configured"):
        BaseLLMClient()


def test_base_llm_client_rejects_unknown_backend_override() -> None:
    caps = BackendCapabilities(
        streaming=True,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    router = LLMRouter(
        [_StubBackend(name="primary", kind="openai_service", default_model="m1", capabilities=caps)]
    )

    with pytest.raises(ValueError, match="Unknown backend"):
        BaseLLMClient(router=router, backend="missing")


def test_base_llm_client_generate_uses_backend_override() -> None:
    caps = BackendCapabilities(
        streaming=True,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    primary = _StubBackend(
        name="primary",
        kind="openai_service",
        default_model="m1",
        capabilities=caps,
    )
    alternate = _StubBackend(
        name="alternate",
        kind="openai_compatible_http",
        default_model="m2",
        capabilities=caps,
    )
    router = LLMRouter([primary, alternate], default_backend="primary")
    client = BaseLLMClient(router=router, backend="alternate")

    response = client.generate(_chat_request(model="m2"))

    assert response.provider == "alternate"
    assert alternate.calls
    assert not primary.calls
    assert alternate.calls[0].metadata["backend"] == "alternate"


def test_base_llm_client_stream_chat_emits_delta_and_completed() -> None:
    caps = BackendCapabilities(
        streaming=True,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    backend = _StubBackend(
        name="streamer",
        kind="openai_service",
        default_model="m1",
        capabilities=caps,
    )
    client = BaseLLMClient(router=LLMRouter([backend]))

    events = list(
        client.stream_chat(
            [LLMMessage(role="user", content="hello")],
            model="m1",
            params=LLMChatParams(),
        )
    )

    assert events[0] == LLMStreamEvent(kind="delta", delta_text="streamer-")
    assert events[1] == LLMStreamEvent(kind="delta", delta_text="stream")
    assert events[2].kind == "completed"
    assert events[2].response is not None
    assert events[2].response.provider == "streamer"


def test_router_prefers_quality_backend_kind() -> None:
    caps = BackendCapabilities(
        streaming=True,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    local = _StubBackend(name="local", kind="llama_cpp", default_model="m-local", capabilities=caps)
    hosted = _StubBackend(
        name="hosted",
        kind="openai_service",
        default_model="m-hosted",
        capabilities=caps,
    )
    router = LLMRouter([local, hosted], default_backend="local")

    response = router.generate(
        LLMRequest(
            messages=[LLMMessage(role="user", content="hello")],
            model="m-hosted",
            temperature=None,
            max_tokens=None,
            tools=(),
            response_schema=None,
            response_format=None,
            metadata={},
            provider_options={},
            task_profile=TaskProfile(priority="quality"),
        )
    )

    assert response.provider == "hosted"


def test_resolve_default_model_uses_configured_router() -> None:
    caps = BackendCapabilities(
        streaming=False,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    backend = _StubBackend(
        name="echo",
        kind="echo_test",
        default_model="echo-model",
        capabilities=caps,
    )
    set_default_router(LLMRouter([backend], default_backend="echo"))

    assert resolve_default_model() == "echo-model"
    assert resolve_default_model(backend="echo") == "echo-model"


def test_router_rejects_unknown_default_backend() -> None:
    caps = BackendCapabilities(
        streaming=True,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    backend = _StubBackend(
        name="echo",
        kind="echo_test",
        default_model="echo-model",
        capabilities=caps,
    )

    with pytest.raises(ValueError, match="Default backend 'missing'"):
        LLMRouter([backend], default_backend="missing")


def test_router_rejects_duplicate_backend_names() -> None:
    caps = BackendCapabilities(
        streaming=True,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    first = _StubBackend(name="dup", kind="echo_test", default_model="m1", capabilities=caps)
    second = _StubBackend(name="dup", kind="openai_service", default_model="m2", capabilities=caps)

    with pytest.raises(ValueError, match="Duplicate backend name 'dup'"):
        LLMRouter([first, second])


def test_router_rejects_unknown_backend_hint() -> None:
    caps = BackendCapabilities(
        streaming=True,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    router = LLMRouter(
        [_StubBackend(name="echo", kind="echo_test", default_model="echo-model", capabilities=caps)]
    )
    request = _chat_request(model="echo-model")
    request = LLMRequest(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        response_schema=request.response_schema,
        response_format=request.response_format,
        metadata={"backend": "missing"},
        provider_options=request.provider_options,
        task_profile=request.task_profile,
    )

    with pytest.raises(LLMCapabilityError, match="Unknown backend 'missing'"):
        router.generate(request)


def test_configure_router_from_yaml_registers_default_router(tmp_path: Path) -> None:
    path = tmp_path / "llm.yaml"
    path.write_text(
        "\n".join(
            [
                "backends:",
                "  - name: echo",
                "    kind: echo_test",
                "    model: echo-v1",
            ]
        ),
        encoding="utf-8",
    )

    router = configure_router_from_yaml(str(path))

    assert isinstance(router, LLMRouter)
    assert resolve_default_model() == "echo-v1"


def test_router_rejects_requests_when_capabilities_missing() -> None:
    caps = BackendCapabilities(
        streaming=False,
        tool_calling="none",
        json_mode="none",
        vision=False,
        max_context_tokens=None,
    )
    backend = _StubBackend(name="basic", kind="echo_test", default_model="m", capabilities=caps)
    router = LLMRouter([backend])

    with pytest.raises(LLMCapabilityError, match="capabilities"):
        router.stream(_chat_request(model="m"))
