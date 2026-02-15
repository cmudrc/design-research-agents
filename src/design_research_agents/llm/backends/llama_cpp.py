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
from design_research_agents.llm.backends.llama_cpp_server import LlamaCppServerBackend
from design_research_agents.llm.backends.utils import messages_to_prompt


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

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            tool_calling="best_effort",
            json_mode="prompt+validate",
            vision=False,
            max_context_tokens=None,
        )

    def healthcheck(self) -> BackendStatus:
        return BackendStatus(ok=True, message="llama.cpp backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        prompt = messages_to_prompt(request.messages)
        text = self._backend.complete(prompt)
        return LLMResponse(text=text, model=request.model, provider=self.name)

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        response = self._generate(request)
        yield LLMDelta(text_delta=response.text)
