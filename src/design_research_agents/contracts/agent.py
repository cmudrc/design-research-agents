"""Agent runtime contracts shared by all concrete agent implementations.

These types define the common output shape, stream event envelope, and protocol
surface that callers can rely on regardless of which agent strategy they use.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from .llm import LLMResponse
from .tools import ToolResult

AgentStreamEventKind = Literal["delta", "completed", "failed"]


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
    """Agent-defined output payload, such as answer content or structured artifacts."""
    success: bool
    """True when the run completed without terminal failure."""
    tool_results: list[ToolResult] = field(default_factory=list)
    """Tool invocation results captured during execution, in call order."""
    model_response: LLMResponse | None = None
    """Final model response associated with the run, when available."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Additional diagnostics, runtime counters, and trace metadata."""

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the result.

        Returns:
            Dictionary representation of the result payload.
        """
        return asdict(self)

    def __str__(self) -> str:
        """Return a JSON-formatted string representation of the result.

        Returns:
            Pretty-printed JSON string for the result.
        """
        return json.dumps(self.asdict(), indent=2, sort_keys=True)

    def __repr__(self) -> str:
        """Return a human-readable string representation of the result.

        Returns:
            Debug-oriented string representation.
        """
        return f"AgentResult({self.asdict()!r})"


@dataclass(slots=True, frozen=True)
class AgentStreamEvent:
    """Single event emitted during streaming agent execution.

    Attributes:
        kind: Event kind indicating partial delta output or terminal completion.
        delta_text: Incremental text payload for ``kind="delta"`` events.
        result: Final :class:`AgentResult` payload for ``kind="completed"``.
    """

    kind: AgentStreamEventKind
    """Event kind indicating delta, completion, or failure."""
    delta_text: str | None = None
    """Incremental text fragment for ``kind='delta'`` events."""
    result: AgentResult | None = None
    """Final result payload for terminal events."""


class Agent(Protocol):
    """Protocol that every agent implementation must satisfy.

    The protocol intentionally keeps the execution contract small: one
    non-streaming call and one streaming call that mirrors the same prompt plus
    explicit runtime options and dependencies.
    """

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one agent run and return the final ``AgentResult`` payload.

        Implementations should treat ``prompt`` as the prompt text for one run.
        Use ``request_id`` and ``dependencies`` for run metadata and upstream
        dependency payloads.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.
        """

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute one agent run and emit stream events through completion.

        Streams must terminate with a ``kind="completed"`` event containing the
        same logical result payload returned by ``run``.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Iterator of streaming events through completion.
        """
