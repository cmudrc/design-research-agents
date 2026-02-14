"""Base LLM client implementation for provider-agnostic workflows."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMProviderAdapter,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents.llm.backends.adapters import build_backend_adapter
from design_research_agents.llm.backends.types import BackendName, parse_backend


class BaseLLMClient(LLMClient):
    """LLM client that delegates generation to configured framework backends."""

    def __init__(self, *, backend: BackendName | None = None) -> None:
        """Initialize a base client bound to one configured backend.

        Args:
            backend: Optional backend name used for client calls. When omitted,
                the current process-wide active backend is used at call time.
        """
        self._backend_override = parse_backend(backend) if backend is not None else None

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate text through the configured backend adapter."""
        # Resolve adapters per-call so runtime reconfiguration takes effect immediately.
        adapter = self._resolve_adapter()
        return adapter.chat(messages, model=model, params=params)

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream response events through the configured backend adapter."""
        # Resolve adapters per-call so runtime reconfiguration takes effect immediately.
        adapter = self._resolve_adapter()
        return adapter.stream_chat(messages, model=model, params=params)

    def _resolve_adapter(self) -> LLMProviderAdapter:
        """Resolve backend adapter with current process-wide runtime configuration."""
        # Import lazily to avoid cyclic imports between llm package entrypoints and client.
        from design_research_agents.llm import (
            _get_active_backend,
            _get_configured_llama_cpp_backend,
            _get_openai_backend_config,
        )

        backend = (
            self._backend_override if self._backend_override is not None else _get_active_backend()
        )
        return build_backend_adapter(
            backend,
            openai_config=_get_openai_backend_config(),
            llama_backend=_get_configured_llama_cpp_backend(),
        )
