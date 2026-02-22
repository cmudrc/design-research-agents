"""Deterministic helpers for design-oriented local examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts import (
    Agent,
    ExecutionResult,
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)


class DeterministicSequenceLLMClient:
    """Small deterministic client for non-networked examples."""

    def __init__(self, *, responses: Sequence[str]) -> None:
        self._responses = list(responses)

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        if not self._responses:
            raise RuntimeError("No deterministic responses remaining.")
        return LLMResponse(model=model, text=self._responses.pop(0), provider="deterministic")

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Sequence[LLMStreamEvent]:
        response = self.chat(messages, model=model, params=params)
        return (
            LLMStreamEvent(kind="delta", delta_text=response.text),
            LLMStreamEvent(kind="completed", response=response),
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.chat(
            request.messages,
            model=request.model or self.default_model(),
            params=LLMChatParams(),
        )

    def stream(self, request: LLMRequest):
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        return "deterministic-model"


class FixedDesignPeerAgent(Agent):
    """Deterministic peer delegate returning fixed design-board contribution."""

    def __init__(self, *, messages: list[str], stop: bool = False) -> None:
        self._messages = list(messages)
        self._stop = stop

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return ExecutionResult(
            output={
                "messages": list(self._messages),
                "proposals": {},
                "decisions": {},
                "stop": self._stop,
            },
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate": "fixed-design-peer"},
        )


class EchoDesignReasoningAgent(Agent):
    """Deterministic local reasoning delegate for RAG examples."""

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del request_id, dependencies
        return ExecutionResult(
            output={
                "summary": "Produced design recommendation from retrieved context.",
                "recommendation": ("Prioritize maintainability checks and explicit testability criteria."),
                "prompt_chars": len(prompt),
            },
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate": "echo-design"},
        )


__all__ = [
    "DeterministicSequenceLLMClient",
    "EchoDesignReasoningAgent",
    "FixedDesignPeerAgent",
]
