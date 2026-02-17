"""Runnable example for ``MultiStepJsonToolCallingAgent`` with local memory."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from design_research_agents import MultiStepJsonToolCallingAgent, Toolbox
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents.contracts.memory import MemoryWriteRecord
from design_research_agents.memory.stores.sqlite_store import SQLiteMemoryStore


class DeterministicLLMClient:
    """Small deterministic LLM stub for non-networked examples."""

    def __init__(self, *, responses: Sequence[str]) -> None:
        """Store deterministic responses returned by chat calls.

        Args:
            responses: Ordered response strings returned by ``chat`` calls.
        """
        self._responses = list(responses)

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Return the next deterministic non-streaming response.

        Args:
            messages: Conversation message history.
            model: Requested model identifier.
            params: Chat parameter bundle.

        Returns:
            Deterministic chat response.

        Raises:
            RuntimeError: Raised when no stub responses remain.
        """
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
    ) -> Iterator[LLMStreamEvent]:
        """Stream one deterministic response as delta then completion.

        Args:
            messages: Conversation message history.
            model: Requested model identifier.
            params: Chat parameter bundle.

        Yields:
            Streaming events for delta text and completion.
        """
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response from a request object.

        Args:
            request: Structured LLM generation request.

        Returns:
            Deterministic LLM response.
        """
        return self.chat(
            request.messages,
            model=request.model or self.default_model(),
            params=LLMChatParams(),
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream one deterministic request-object response.

        Args:
            request: Structured LLM generation request.

        Yields:
            Deterministic delta chunks for the generated text.
        """
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        """Return deterministic default model name.

        Returns:
            Deterministic model identifier.
        """
        return "deterministic-model"


def main() -> None:
    """Run one multi-step JSON tool call with memory retrieval and write-back."""
    store = SQLiteMemoryStore()
    store.write(
        [MemoryWriteRecord(content="Prior note: use calculator for arithmetic tasks.")],
        namespace="examples",
    )

    llm_client = DeterministicLLMClient(
        responses=[
            '{"continue": true, "thought": "start"}',
            '{"tool_name": "calculator", "tool_input": {"expression": "12 * (4 + 1)"}}',
            '{"continue": false, "thought": "done"}',
        ]
    )
    agent = MultiStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
        memory_store=store,
        memory_namespace="examples",
        memory_read_top_k=3,
        memory_write_observations=True,
    )
    result = agent.run("Compute 12 * (4 + 1).")
    print(result.asdict())
    store.close()


if __name__ == "__main__":
    main()
