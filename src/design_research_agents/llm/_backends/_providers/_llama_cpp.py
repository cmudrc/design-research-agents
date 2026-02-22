"""llama.cpp backend wrapper using the managed server."""

from __future__ import annotations

from collections.abc import Iterator

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm._backends._base import BaseLLMBackend
from design_research_agents.llm._backends._providers._llama_cpp_server import (
    LlamaCppServerBackend,
)
from design_research_agents.llm._backends._providers._openai_compatible_http import (
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
            name: Input value for this parameter.
            llama_backend: Input value for this parameter.
            default_model: Input value for this parameter.
            config_hash: Input value for this parameter.
            max_retries: Input value for this parameter.
            model_patterns: Input value for this parameter.
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
            Computed return value.
        """
        return _LLAMA_CPP_CAPABILITIES

    def healthcheck(self) -> BackendStatus:
        """Return static health status for configured llama.cpp backend.

        Returns:
            Computed return value.
        """
        return BackendStatus(ok=True, message="llama.cpp backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response via the managed llama.cpp HTTP backend.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.
        """
        self._backend.start()
        return self._http_backend.generate(request)

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream one response delta via the managed llama.cpp backend.

        Args:
            request: Input value for this parameter.

        Yields:
            The yielded values.
        """
        response = self._generate(request)
        yield LLMDelta(text_delta=response.text)
