"""Shared deterministic helpers for streaming agent examples."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Sequence

from design_research_agents.contracts.agent import AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMMessage,
    LLMResponse,
    LLMStreamEvent,
)


class StaticResponseLLMClient:
    """LLM stub that always returns one configured response text."""

    def __init__(self, *, response_text: str, provider: str = "example-stub") -> None:
        self._response_text = response_text
        self._provider = provider

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Return one static response payload.

        Args:
            messages: Provider-neutral chat message sequence.
            model: Model identifier for the configured backend.
            params: Provider-neutral generation parameters.

        Returns:
            Normalized response payload with the configured text.
        """
        del messages, params
        return LLMResponse(model=model, text=self._response_text, provider=self._provider)

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Emit one delta event followed by a completion event.

        Args:
            messages: Provider-neutral chat message sequence.
            model: Model identifier for the configured backend.
            params: Provider-neutral generation parameters.

        Yields:
            Streaming events for the configured response.
        """
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)


class SequenceResponseLLMClient:
    """LLM stub that returns a deterministic sequence of response texts."""

    def __init__(self, *, response_texts: Sequence[str], provider: str = "example-stub") -> None:
        self._response_texts = list(response_texts)
        self._provider = provider

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Return the next configured response payload.

        Args:
            messages: Provider-neutral chat message sequence.
            model: Model identifier for the configured backend.
            params: Provider-neutral generation parameters.

        Returns:
            Normalized response payload with the next configured text.
        """
        del messages, params
        if not self._response_texts:
            raise ValueError("No configured responses remaining in SequenceResponseLLMClient.")
        return LLMResponse(
            model=model,
            text=self._response_texts.pop(0),
            provider=self._provider,
        )

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Emit one delta event followed by a completion event.

        Args:
            messages: Provider-neutral chat message sequence.
            model: Model identifier for the configured backend.
            params: Provider-neutral generation parameters.

        Yields:
            Streaming events for the next configured response.
        """
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)


def print_stream_event(event: AgentStreamEvent) -> None:
    """Print one agent stream event with stable prefixes for quick inspection.

    Args:
        event: Stream event payload to render.
    """
    if event.kind == "delta":
        print(f"delta: {event.delta_text or ''}")
        return

    if event.result is None:
        print("completed: null")
        return

    print("completed:")
    print(json.dumps(dataclasses.asdict(event.result), indent=2, sort_keys=True))
