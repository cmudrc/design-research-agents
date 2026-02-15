"""Base LLM client implementation for legacy adapters and router backends."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMDelta,
    LLMMessage,
    LLMProviderAdapter,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents.llm.backends.adapters import build_backend_adapter
from design_research_agents.llm.backends.types import BackendName, parse_backend
from design_research_agents.llm.router import LLMRouter


class BaseLLMClient(LLMClient):
    """LLM client that supports both legacy adapters and router dispatch."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        router: LLMRouter | None = None,
    ) -> None:
        resolved_router = router or _resolve_default_router()
        self._router = resolved_router
        self._backend_override: str | None = _normalize_backend_override(backend)
        if resolved_router is None:
            if self._backend_override is not None:
                self._legacy_backend_override = parse_backend(self._backend_override)
            else:
                self._legacy_backend_override = None
        else:
            self._legacy_backend_override = None

    @classmethod
    def from_openai(
        cls,
        *,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        base_url: str | None = None,
        require_api_key: bool = True,
    ) -> BaseLLMClient:
        """Configure OpenAI defaults and return an OpenAI-bound client."""
        from design_research_agents.llm import configure_openai

        configure_openai(
            model=model,
            api_key_env=api_key_env,
            api_key=api_key,
            base_url=base_url,
            require_api_key=require_api_key,
        )
        return cls(backend="openai")

    @classmethod
    def from_llama_cpp_server(
        cls,
        model: str,
        *,
        hf_model_repo_id: str | None = None,
        api_model: str = "local-model",
        host: str = "127.0.0.1",
        port: int = 8001,
        startup_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
        extra_server_args: Sequence[str] = (),
    ) -> BaseLLMClient:
        """Configure llama-cpp defaults and return a llama-bound client."""
        from design_research_agents.llm import configure_llama_cpp_server

        configure_llama_cpp_server(
            model=model,
            hf_model_repo_id=hf_model_repo_id,
            api_model=api_model,
            host=host,
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            extra_server_args=extra_server_args,
        )
        return cls(backend="llama-cpp-server")

    @classmethod
    def from_transformers(
        cls,
        *,
        model: str,
        tokenizer: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        device: int | str | None = None,
        device_map: str | None = None,
        torch_dtype: object | None = None,
        cache_dir: str | None = None,
        use_fast: bool = True,
        pipeline_task: str = "text-generation",
        model_kwargs: dict[str, object] | None = None,
        tokenizer_kwargs: dict[str, object] | None = None,
        pipeline_kwargs: dict[str, object] | None = None,
        generation_kwargs: dict[str, object] | None = None,
    ) -> BaseLLMClient:
        """Configure Transformers defaults and return a pinned client."""
        from design_research_agents.llm import configure_transformers

        configure_transformers(
            model=model,
            tokenizer=tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            use_fast=use_fast,
            pipeline_task=pipeline_task,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            pipeline_kwargs=pipeline_kwargs,
            generation_kwargs=generation_kwargs,
        )
        return cls(backend="transformers")

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._router is not None:
            routed_request = _with_backend_override(request, self._backend_override)
            return self._router.generate(routed_request)

        adapter = self._resolve_adapter()
        model = request.model or self.default_model()
        response = adapter.chat(
            request.messages,
            model=model,
            params=LLMChatParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_schema=request.response_schema,
                provider_options=dict(request.provider_options),
            ),
        )
        return _ensure_response_model(response, model=model)

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        if self._router is None:
            adapter = self._resolve_adapter()
            response = adapter.chat(messages, model=model, params=params)
            return _ensure_response_model(response, model=model)

        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            response_schema=params.response_schema,
            provider_options=dict(params.provider_options),
        )
        return self.generate(request)

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        if self._router is not None:
            routed_request = _with_backend_override(request, self._backend_override)
            return self._router.stream(routed_request)

        model = request.model or self.default_model()
        params = LLMChatParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_schema=request.response_schema,
            provider_options=dict(request.provider_options),
        )

        def _iterator() -> Iterator[LLMDelta]:
            for event in self.stream_chat(request.messages, model=model, params=params):
                if event.kind == "delta":
                    yield LLMDelta(text_delta=event.delta_text or "")

        return _iterator()

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        if self._router is None:
            adapter = self._resolve_adapter()
            yield from adapter.stream_chat(messages, model=model, params=params)
            return

        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            response_schema=params.response_schema,
            provider_options=dict(params.provider_options),
        )
        stream = self.stream(request)
        accumulated_text = ""
        completed_response: LLMResponse | None = None
        for delta in stream:
            if delta.text_delta:
                accumulated_text += delta.text_delta
                yield LLMStreamEvent(kind="delta", delta_text=delta.text_delta)
        response = getattr(stream, "response", None)
        if isinstance(response, LLMResponse):
            completed_response = response
        if completed_response is None and accumulated_text:
            completed_response = LLMResponse(model=model, text=accumulated_text)
        if completed_response is None:
            completed_response = self.generate(request)
        yield LLMStreamEvent(kind="completed", response=completed_response)

    def default_model(self) -> str:
        if self._router is None:
            from design_research_agents.llm import resolve_default_model

            return resolve_default_model(
                backend=cast(str | None, self._legacy_backend_override),
            )

        if self._backend_override:
            backend_map = getattr(self._router, "_backend_map", {})
            backend = backend_map.get(self._backend_override)
            model = getattr(backend, "default_model", None) if backend is not None else None
            if isinstance(model, str) and model:
                return model
        return self._router.default_model()

    def _resolve_adapter(self) -> LLMProviderAdapter:
        from design_research_agents.llm import (
            _get_active_backend,
            _get_configured_llama_cpp_backend,
            _get_configured_transformers_backend,
            _get_openai_backend_config,
        )

        backend: BackendName
        if self._legacy_backend_override is not None:
            backend = self._legacy_backend_override
        else:
            backend = _get_active_backend()
        return build_backend_adapter(
            backend,
            openai_config=_get_openai_backend_config(),
            llama_backend=_get_configured_llama_cpp_backend(),
            transformers_backend=_get_configured_transformers_backend(),
        )


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


def _ensure_response_model(response: LLMResponse, *, model: str) -> LLMResponse:
    if response.model:
        return response
    return LLMResponse(
        text=response.text,
        model=model,
        provider=response.provider,
        finish_reason=response.finish_reason,
        usage=response.usage,
        latency_ms=response.latency_ms,
        raw_output=response.raw_output,
        tool_calls=response.tool_calls,
        raw=response.raw,
        provenance=response.provenance,
    )
