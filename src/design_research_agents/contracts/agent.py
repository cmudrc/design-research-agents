"""Agent runtime contracts shared by all concrete agent implementations.

These types define the common output shape, stream event envelope, and protocol
surface that callers can rely on regardless of which agent strategy they use.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .llm import LLMResponse
from .tools import ToolResult

AgentStreamEventKind = Literal["delta", "completed"]
AgentInput = Mapping[str, object] | str


@dataclass(slots=True, frozen=True)
class AgentResult:
    """Structured output produced by one agent execution.

    Attributes:
        output: Agent-specific payload containing final data or error details.
        success: Boolean success flag for the overall run.
        tool_results: Ordered list of tool invocations performed during the run.
        model_response: Optional model response associated with the run.
        metadata: Additional diagnostics and trace metadata for callers.
    """

    output: dict[str, object]
    success: bool
    tool_results: list[ToolResult] = field(default_factory=list)
    model_response: LLMResponse | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentStreamEvent:
    """Single event emitted during streaming agent execution.

    Attributes:
        kind: Event kind indicating partial delta output or terminal completion.
        delta_text: Incremental text payload for ``kind="delta"`` events.
        result: Final :class:`AgentResult` payload for ``kind="completed"``.
    """

    kind: AgentStreamEventKind
    delta_text: str | None = None
    result: AgentResult | None = None


class Agent(Protocol):
    """Protocol that every agent implementation must satisfy.

    The protocol intentionally keeps the execution contract small: one
    non-streaming call and one streaming call that mirrors the same input plus
    explicit runtime options and dependencies.
    """

    def run(
        self,
        input: AgentInput,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one agent run and return the final ``AgentResult`` payload.

        Implementations should treat ``input`` as run-specific request data.
        Callers may provide a mapping payload or plain string shorthand
        (interpreted as ``{"prompt": <input>}``). Use ``request_id`` and
        ``dependencies`` for run metadata and upstream dependency payloads.
        """

    def run_stream(
        self,
        input: AgentInput,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute one agent run and emit stream events through completion.

        Streams must terminate with a ``kind="completed"`` event containing the
        same logical result payload returned by ``run``.
        """
