"""llama.cpp backend wrapper using the managed server."""

from __future__ import annotations

from collections.abc import Iterator

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm.backends.base import BaseLLMBackend
from design_research_agents.llm.backends.providers.llama_cpp_server import (
    LlamaCppServerBackend,
)
from design_research_agents.llm.backends.providers.openai_compatible_http import (
    OpenAICompatibleHTTPBackend,
)

_LLAMA_CPP_CAPABILITIES = BackendCapabilities(
    streaming=True,
    tool_calling="best_effort",
    json_mode="prompt+validate",
    vision=False,
    max_context_tokens=None,
)


class LlamaCppBackend(BaseLLMBackend):
    """Local llama.cpp backend using the managed OpenAI-compatible server."""

    def __init__(
        self,
        *,
        name: str,
        llama_backend: LlamaCppServerBackend,
        default_model: str,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] = (),
    ) -> None:
        """Initialize the managed llama.cpp server wrapper backend.

        Args:
            name: Parameter value.
            llama_backend: Parameter value.
            default_model: Parameter value.
            config_hash: Parameter value.
            max_retries: Parameter value.
            model_patterns: Parameter value.
        """
        super().__init__(
            name=name,
            kind="llama_cpp",
            default_model=default_model,
            base_url=llama_backend.base_url,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self._backend = llama_backend
        self._http_backend = OpenAICompatibleHTTPBackend(
            name=name,
            base_url=llama_backend.base_url,
            default_model=default_model,
            api_key_env="LLAMA_CPP_SERVER_API_KEY",
            api_key=None,
            capabilities=_LLAMA_CPP_CAPABILITIES,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )

    def capabilities(self) -> BackendCapabilities:
        """Return capabilities provided by the wrapped llama.cpp server.

        Returns:
            The resulting value.
        """
        return _LLAMA_CPP_CAPABILITIES

    def healthcheck(self) -> BackendStatus:
        """Return static health status for configured llama.cpp backend.

        Returns:
            The resulting value.
        """
        return BackendStatus(ok=True, message="llama.cpp backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Run generate.

        Args:
            request: Parameter value.

        Returns:
            The resulting value.
        """
        self._backend.start()
        return self._http_backend.generate(request)

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Run stream.

        Args:
            request: Parameter value.

        Yields:
            The yielded values.
        """
        response = self._generate(request)
        yield LLMDelta(text_delta=response.text)
