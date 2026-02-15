"""Router-first LLM client implementation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents.llm.router import LLMRouter


class BaseLLMClient(LLMClient):
    """Provider-agnostic client that delegates to an ``LLMRouter``."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        router: LLMRouter | None = None,
    ) -> None:
        resolved_router = router or _resolve_default_router()
        if resolved_router is None:
            raise ValueError(
                "No LLM router configured. Pass router=... or call "
                "design_research_agents.llm.configure_router_from_yaml(...)."
            )
        self._router = resolved_router
        self._backend_override = _normalize_backend_override(backend)
        if (
            self._backend_override is not None
            and self._router.backend(self._backend_override) is None
        ):
            raise ValueError(
                f"Unknown backend '{self._backend_override}'. "
                f"Configured backends: {', '.join(self._router.backend_names()) or '<none>'}."
            )

    def generate(self, request: LLMRequest) -> LLMResponse:
        routed_request = _with_backend_override(request, self._backend_override)
        return self._router.generate(routed_request)

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            tools=(),
            response_schema=params.response_schema,
            response_format=None,
            metadata={},
            provider_options=dict(params.provider_options),
            task_profile=None,
        )
        return self.generate(request)

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        routed_request = _with_backend_override(request, self._backend_override)
        return self._router.stream(routed_request)

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            tools=(),
            response_schema=params.response_schema,
            response_format=None,
            metadata={},
            provider_options=dict(params.provider_options),
            task_profile=None,
        )
        stream = self.stream(request)
        full_text = ""
        for delta in stream:
            if delta.text_delta:
                full_text += delta.text_delta
                yield LLMStreamEvent(kind="delta", delta_text=delta.text_delta)
        completed = getattr(stream, "response", None)
        if not isinstance(completed, LLMResponse):
            completed = LLMResponse(
                text=full_text,
                model=model,
                provider=None,
                finish_reason=None,
                usage=None,
                latency_ms=None,
                raw_output=None,
                tool_calls=(),
                raw=None,
                provenance=None,
            )
        yield LLMStreamEvent(kind="completed", response=completed)

    def default_model(self) -> str:
        if self._backend_override:
            return self._router.default_model_for_backend(self._backend_override)
        return self._router.default_model()


def _resolve_default_router() -> LLMRouter | None:
    from design_research_agents.llm import _get_default_router

    return _get_default_router()


def _normalize_backend_override(backend: str | None) -> str | None:
    if backend is None:
        return None
    normalized = backend.strip()
    return normalized or None


def _with_backend_override(request: LLMRequest, backend: str | None) -> LLMRequest:
    if backend is None:
        return request
    metadata = dict(request.metadata)
    metadata["backend"] = backend
    return LLMRequest(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        response_schema=request.response_schema,
        response_format=request.response_format,
        metadata=metadata,
        provider_options=dict(request.provider_options),
        task_profile=request.task_profile,
    )
