"""Deterministic echo backend used for tests and smoke checks."""

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
from design_research_agents.llm.backends.utils import messages_to_prompt


class EchoTestBackend(BaseLLMBackend):
    """Echo backend that returns a normalized prompt string."""

    def __init__(self, *, name: str, model: str, config_hash: str) -> None:
        super().__init__(
            name=name,
            kind="echo_test",
            default_model=model,
            base_url=None,
            config_hash=config_hash,
            max_retries=0,
            model_patterns=(model,),
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            tool_calling="none",
            json_mode="none",
            vision=False,
            max_context_tokens=None,
        )

    def healthcheck(self) -> BackendStatus:
        return BackendStatus(ok=True, message="echo-test backend is always healthy.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        prompt = messages_to_prompt(request.messages)
        cleaned_prompt = " ".join(prompt.strip().split())
        if not cleaned_prompt:
            cleaned_prompt = "Hello from design-research-agents."
        text = f"[{request.model}] {cleaned_prompt}"
        return LLMResponse(text=text, model=request.model, provider=self.name)

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        response = self._generate(request)
        yield LLMDelta(text_delta=response.text)
