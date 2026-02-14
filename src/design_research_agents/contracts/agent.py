"""Agent runtime contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .llm import LLMResponse
from .tools import ToolResult

AgentStreamEventKind = Literal["delta", "completed"]


@dataclass(slots=True, frozen=True)
class AgentResult:
    """Structured output from an agent execution."""

    output: dict[str, object]
    success: bool
    tool_results: list[ToolResult] = field(default_factory=list)
    model_response: LLMResponse | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentStreamEvent:
    """Single event emitted during streaming agent execution."""

    kind: AgentStreamEventKind
    delta_text: str | None = None
    result: AgentResult | None = None


class Agent(Protocol):
    """Protocol that all agents must implement."""

    def run(self, input: Mapping[str, object], context: Mapping[str, object]) -> AgentResult:
        """Execute the agent for one input/context pair."""

    def run_stream(
        self,
        input: Mapping[str, object],
        context: Mapping[str, object],
    ) -> Iterator[AgentStreamEvent]:
        """Execute the agent and emit streamed events."""
