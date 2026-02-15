"""Shared deterministic helpers for basic agent examples."""

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


class RecordingSequenceLLMClient:
    """Deterministic LLM stub that records invocation style."""

    def __init__(self, *, response_texts: Sequence[str], provider: str = "example-stub") -> None:
        self._responses = list(response_texts)
        self._provider = provider
        self.chat_calls = 0
        self.generate_calls = 0
        self.stream_calls = 0
        self.stream_chat_calls = 0

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        self.chat_calls += 1
        if not self._responses:
            raise ValueError("No configured responses remaining.")
        return LLMResponse(model=model, text=self._responses.pop(0), provider=self._provider)

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        self.stream_chat_calls += 1
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        if not self._responses:
            raise ValueError("No configured responses remaining.")
        return LLMResponse(
            model=request.model or self.default_model(),
            text=self._responses.pop(0),
            provider=self._provider,
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        self.stream_calls += 1
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        return "example-model"

    def assert_exhausted(self) -> None:
        """Raise when responses remain unused."""
        if self._responses:
            raise ValueError(f"Unused stub responses remaining: {len(self._responses)}")
