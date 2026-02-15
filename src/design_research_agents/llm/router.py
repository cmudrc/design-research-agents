"""Backend router that selects LLM backends per request."""

from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from design_research_agents.contracts.llm import (
    LLMCapabilityError,
    LLMChatParams,
    LLMClient,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    Provenance,
    TaskProfile,
    ToolCall,
    Usage,
)
from design_research_agents.llm.backends.base import BaseLLMBackend

_LOGGER = logging.getLogger("design_research_agents.llm")


@dataclass(slots=True)
class StreamAccumulator:
    """Accumulate deltas into a final response."""

    text_parts: list[str] = field(default_factory=list)
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage: Usage | None = None

    def apply(self, delta: LLMDelta) -> None:
        """Apply one LLM delta into accumulated text/tool/usage state."""
        if delta.text_delta:
            self.text_parts.append(delta.text_delta)
        if delta.tool_call_delta:
            call_id = delta.tool_call_delta.call_id or "call_1"
            entry = self.tool_calls.setdefault(
                call_id,
                {
                    "name": delta.tool_call_delta.name,
                    "arguments": "",
                },
            )
            if delta.tool_call_delta.name:
                entry["name"] = delta.tool_call_delta.name
            if delta.tool_call_delta.arguments_json_delta:
                entry["arguments"] += delta.tool_call_delta.arguments_json_delta
        if delta.usage_delta:
            self.usage = delta.usage_delta

    def build_tool_calls(self) -> tuple[ToolCall, ...]:
        """Build normalized tool calls from accumulated deltas."""
        calls: list[ToolCall] = []
        for call_id, payload in self.tool_calls.items():
            name = payload.get("name") or ""
            arguments = payload.get("arguments") or ""
            calls.append(ToolCall(name=name, arguments_json=arguments, call_id=call_id))
        return tuple(calls)

    def text(self) -> str:
        """Return accumulated plain-text output."""
        return "".join(self.text_parts)


class LLMStream(Iterator[LLMDelta]):
    """Iterator wrapper that exposes a final response after streaming."""

    def __init__(
        self,
        iterator: Iterator[LLMDelta],
        accumulator: StreamAccumulator,
    ) -> None:
        """Wrap a delta iterator and expose a completed response later."""
        self._iterator = iterator
        self._accumulator = accumulator
        self.response: LLMResponse | None = None

    def __iter__(self) -> LLMStream:
        """Return this stream iterator instance."""
        return self

    def __next__(self) -> LLMDelta:
        """Yield the next streaming delta."""
        return next(self._iterator)

    @property
    def accumulator(self) -> StreamAccumulator:
        """Return the accumulator attached to this stream."""
        return self._accumulator


class LLMRouter(LLMClient):
    """Route each request to the best backend given constraints and profile."""

    def __init__(
        self,
        backends: Sequence[BaseLLMBackend],
        *,
        default_backend: str | None = None,
    ) -> None:
        """Initialize backend routing with optional default backend override."""
        if not backends:
            raise ValueError("At least one backend must be configured.")
        seen_names: set[str] = set()
        for backend in backends:
            if backend.name in seen_names:
                raise ValueError(f"Duplicate backend name '{backend.name}'.")
            seen_names.add(backend.name)
        self._backends = list(backends)
        self._backend_map = {backend.name: backend for backend in backends}
        self._default_backend = default_backend
        if self._default_backend and self._default_backend not in self._backend_map:
            raise ValueError(f"Default backend '{self._default_backend}' is not configured.")

    def backend_names(self) -> tuple[str, ...]:
        """Return configured backend names in deterministic order."""
        return tuple(backend.name for backend in self._backends)

    def backend(self, name: str) -> BaseLLMBackend | None:
        """Return a backend by name, or ``None`` when missing."""
        return self._backend_map.get(name)

    def default_model_for_backend(self, name: str) -> str:
        """Return the default model for a specific named backend."""
        backend = self.backend(name)
        if backend is None:
            raise ValueError(f"Unknown backend '{name}'.")
        if backend.default_model:
            return backend.default_model
        raise ValueError(f"Backend '{name}' does not define a default model.")

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response by selecting the best matching backend."""
        backend = self._select_backend(request, require_streaming=False)
        started_at = Provenance.now_iso()
        start_time = time.perf_counter()
        response = backend.generate(request)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        completed_at = Provenance.now_iso()
        response_with_meta = _attach_provenance(
            response=response,
            backend=backend,
            request=request,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
        )
        _log_request(
            request=request,
            response=response_with_meta,
            backend=backend,
            latency_ms=latency_ms,
            retries=_extract_retries(response_with_meta),
        )
        return response_with_meta

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Build and execute a request-object call from chat-style inputs."""
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
        """Stream response deltas from the selected backend."""
        backend = self._select_backend(request, require_streaming=True)
        started_at = Provenance.now_iso()
        start_time = time.perf_counter()
        accumulator = StreamAccumulator()

        def _iterator() -> Iterator[LLMDelta]:
            try:
                for delta in backend.stream(request):
                    accumulator.apply(delta)
                    yield delta
            finally:
                completed_at = Provenance.now_iso()
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                response = LLMResponse(
                    text=accumulator.text(),
                    tool_calls=accumulator.build_tool_calls(),
                    usage=accumulator.usage,
                    model=request.model or backend.default_model,
                    provider=backend.name,
                    latency_ms=latency_ms,
                )
                response_with_meta = _attach_provenance(
                    response=response,
                    backend=backend,
                    request=request,
                    started_at=started_at,
                    completed_at=completed_at,
                    latency_ms=latency_ms,
                )
                _log_request(
                    request=request,
                    response=response_with_meta,
                    backend=backend,
                    latency_ms=latency_ms,
                    retries=None,
                )
                stream.response = response_with_meta

        stream = LLMStream(_iterator(), accumulator)
        return stream

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Build and execute a streaming request from chat-style inputs."""
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
        """Return the default model of the resolved default backend."""
        backend = self._resolve_default_backend()
        if backend.default_model:
            return backend.default_model
        raise ValueError("Default backend does not specify a default model.")

    def _resolve_default_backend(self) -> BaseLLMBackend:
        if self._default_backend:
            backend = self._backend_map.get(self._default_backend)
            if backend:
                return backend
        return self._backends[0]

    def _select_backend(self, request: LLMRequest, *, require_streaming: bool) -> BaseLLMBackend:
        backend_hint = request.metadata.get("backend") if request.metadata else None
        if isinstance(backend_hint, str):
            backend = self.backend(backend_hint)
            if backend is None:
                raise LLMCapabilityError(f"Unknown backend '{backend_hint}'.")
            _ensure_backend_supports(backend, request, require_streaming=require_streaming)
            return backend

        kind_hint = request.metadata.get("backend_kind") if request.metadata else None
        if isinstance(kind_hint, str):
            candidates = [
                backend
                for backend in self._backends
                if backend.kind == kind_hint
                and _backend_matches_request(backend, request, require_streaming=require_streaming)
            ]
            if candidates:
                return candidates[0]

        candidates = [
            backend
            for backend in self._backends
            if _backend_matches_request(backend, request, require_streaming=require_streaming)
        ]
        if not candidates:
            raise LLMCapabilityError("No backend satisfies the requested capabilities.")
        profile = request.task_profile or TaskProfile()
        priority_kinds = _priority_kinds(profile)
        for kind in priority_kinds:
            for backend in candidates:
                if backend.kind == kind:
                    return backend
        return candidates[0]


def _attach_provenance(
    *,
    response: LLMResponse,
    backend: BaseLLMBackend,
    request: LLMRequest,
    started_at: str,
    completed_at: str,
    latency_ms: int,
) -> LLMResponse:
    model_id = response.model or request.model or backend.default_model or ""
    provenance = Provenance(
        backend_name=backend.name,
        backend_kind=backend.kind,
        model_id=model_id,
        base_url=backend.base_url,
        started_at=started_at,
        completed_at=completed_at,
        config_hash=backend.config_hash,
    )
    return LLMResponse(
        text=response.text,
        tool_calls=response.tool_calls,
        usage=response.usage,
        raw=response.raw,
        provenance=provenance,
        model=model_id,
        provider=backend.name,
        finish_reason=response.finish_reason,
        latency_ms=latency_ms if response.latency_ms is None else response.latency_ms,
    )


def _backend_matches_request(
    backend: BaseLLMBackend,
    request: LLMRequest,
    *,
    require_streaming: bool,
) -> bool:
    capabilities = backend.capabilities()
    if require_streaming and not capabilities.streaming:
        return False
    if request.tools and capabilities.tool_calling == "none":
        return False
    if (request.response_schema or request.response_format) and capabilities.json_mode == "none":
        return False
    if request.model and not backend.supports_model(request.model):
        return False
    return not (backend.kind == "mlx_local" and not _is_macos_arm())


def _ensure_backend_supports(
    backend: BaseLLMBackend,
    request: LLMRequest,
    *,
    require_streaming: bool,
) -> None:
    if not _backend_matches_request(backend, request, require_streaming=require_streaming):
        raise LLMCapabilityError(
            f"Backend '{backend.name}' does not satisfy required capabilities."
        )


def _priority_kinds(profile: TaskProfile) -> tuple[str, ...]:
    tags = set(profile.tags)
    if "interactive" in tags or profile.priority == "latency":
        if _is_macos_arm():
            return (
                "mlx_local",
                "llama_cpp",
                "transformers_local",
                "openai_compatible_http",
                "openai_service",
            )
        return (
            "llama_cpp",
            "transformers_local",
            "openai_compatible_http",
            "openai_service",
        )
    if "heavy" in tags:
        return (
            "openai_compatible_http",
            "openai_service",
            "transformers_local",
            "mlx_local",
            "llama_cpp",
        )
    if "reliability" in tags or profile.priority == "quality":
        return (
            "openai_service",
            "openai_compatible_http",
            "transformers_local",
            "mlx_local",
            "llama_cpp",
        )
    if profile.priority == "cost":
        return (
            "llama_cpp",
            "mlx_local",
            "transformers_local",
            "openai_compatible_http",
            "openai_service",
        )
    return (
        "mlx_local",
        "llama_cpp",
        "openai_compatible_http",
        "openai_service",
        "transformers_local",
    )


def _is_macos_arm() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower().startswith("arm")


def _extract_retries(response: LLMResponse) -> int | None:
    raw = response.raw or {}
    structured = raw.get("structured_output")
    if isinstance(structured, dict):
        attempts = structured.get("attempts")
        if isinstance(attempts, int):
            return max(0, attempts - 1)
    return None


def _log_request(
    *,
    request: LLMRequest,
    response: LLMResponse,
    backend: BaseLLMBackend,
    latency_ms: int,
    retries: int | None,
) -> None:
    request_id = None
    if request.metadata:
        request_id = request.metadata.get("request_id") or request.metadata.get("trace_id")
    tool_calls = [call.name for call in response.tool_calls]
    payload = {
        "request_id": request_id,
        "backend": backend.name,
        "backend_kind": backend.kind,
        "model": response.model,
        "latency_ms": latency_ms,
        "usage": _usage_dict(response.usage),
        "retries": retries,
        "tool_calls": tool_calls,
    }
    try:
        _LOGGER.info(json.dumps(payload))
    except Exception:
        _LOGGER.info("LLM request completed: %s", payload)


def _usage_dict(usage: Usage | dict[str, int] | None) -> dict[str, int] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return {
            "prompt_tokens": _coerce_usage_value(usage.get("prompt_tokens")),
            "completion_tokens": _coerce_usage_value(usage.get("completion_tokens")),
            "total_tokens": _coerce_usage_value(usage.get("total_tokens")),
        }
    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }


def _coerce_usage_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
