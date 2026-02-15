"""Shared deterministic helpers for runtime and workflow basic examples."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)


class SequenceResponseLLMClient:
    """Deterministic LLM stub that returns configured responses in order."""

    def __init__(self, *, response_texts: Sequence[str]) -> None:
        self._responses = list(response_texts)

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        if not self._responses:
            raise ValueError("No configured responses remaining.")
        return LLMResponse(model=model, text=self._responses.pop(0), provider="example-stub")

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Compatibility helper for request-object clients."""
        return self.chat(
            request.messages,
            model=request.model or self.default_model(),
            params=LLMChatParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_schema=request.response_schema,
                provider_options=dict(request.provider_options),
            ),
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Compatibility helper for request-object streaming clients."""
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        return "example-model"
