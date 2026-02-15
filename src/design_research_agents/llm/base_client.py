"""Base LLM client implementation for provider-agnostic workflows.

The client resolves backend adapters at call time so process-level
reconfiguration is reflected immediately without recreating the client.
"""

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
    """LLM client that delegates generation to configured backend adapters.

    A client can pin a backend override or inherit whichever backend is
    currently active in process-wide configuration. Convenience constructors
    cover the common "configure backend + create client" setup flow.
    """

    def __init__(self, *, backend: BackendName | None = None) -> None:
        """Initialize a base client bound to one configured backend.

        Args:
            backend: Optional backend name used for client calls. When omitted,
                the current process-wide active backend is used at call time.
        """
        self._backend_override = parse_backend(backend) if backend is not None else None

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
        """Configure OpenAI defaults and return an OpenAI-bound client.

        This applies process-wide OpenAI configuration and returns a client
        pinned to ``backend="openai"``.

        Args:
            model: OpenAI model name used by default.
            api_key_env: Environment variable name used for API key lookup.
            api_key: Explicit API key value. When provided, it overrides ``api_key_env``.
            base_url: Optional OpenAI-compatible API base URL.
            require_api_key: Whether missing API keys should raise an error.

        Returns:
            Base client pinned to the OpenAI backend.
        """
        # Import lazily to avoid cyclic imports between llm package entrypoints and client.
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
        """Configure llama-cpp-server defaults and return a llama-bound client.

        This applies process-wide llama-cpp-server configuration and returns a
        client pinned to ``backend="llama-cpp-server"``.

        Args:
            model: ``llama_cpp.server`` ``--model`` value.
            hf_model_repo_id: Optional Hugging Face repository id.
            api_model: OpenAI-compatible model identifier used for completions.
            host: Host used by the local server.
            port: Port used by the local server.
            startup_timeout_seconds: Max startup wait duration.
            poll_interval_seconds: Delay between readiness checks.
            extra_server_args: Extra CLI arguments for ``llama_cpp.server``.

        Returns:
            Base client pinned to the llama-cpp-server backend.
        """
        # Import lazily to avoid cyclic imports between llm package entrypoints and client.
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

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate full chat response through resolved backend adapter.

        Adapter resolution happens at call time so global config changes apply.

        Args:
            messages: Provider-neutral chat message sequence.
            model: Model identifier for the configured backend.
            params: Provider-neutral generation parameters.

        Returns:
            Normalized response payload from the backend adapter.
        """
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
        """Stream chat response events through resolved backend adapter.

        Adapter resolution happens at call time so global config changes apply.

        Args:
            messages: Provider-neutral chat message sequence.
            model: Model identifier for the configured backend.
            params: Provider-neutral generation parameters.

        Returns:
            Iterator over normalized streaming events.
        """
        # Resolve adapters per-call so runtime reconfiguration takes effect immediately.
        adapter = self._resolve_adapter()
        return adapter.stream_chat(messages, model=model, params=params)

    def _resolve_adapter(self) -> LLMProviderAdapter:
        """Resolve backend adapter using current runtime configuration.

        If no client override exists, process-wide active backend is used.

        Returns:
            Backend adapter for the resolved backend.
        """
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

    def default_model(self) -> str:
        """Resolve default model name for this client's effective backend.

        This helper mirrors the same backend-override semantics used by ``chat``.

        Returns:
            Default model identifier for the effective backend.
        """
        # Import lazily to avoid cyclic imports between llm package entrypoints and client.
        from design_research_agents.llm import resolve_default_model

        return resolve_default_model(backend=self._backend_override)
