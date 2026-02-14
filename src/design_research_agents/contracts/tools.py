"""Tool specification payloads and runtime protocol contracts.

These definitions describe how tools are registered, invoked, and reported
across agents and runtimes in a provider-neutral manner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True, frozen=True)
class ToolCostHints:
    """Approximate cost metadata associated with a tool invocation.

    Attributes:
        token_cost_estimate: Estimated token cost consumed by the tool.
        latency_ms_estimate: Estimated wall-clock latency in milliseconds.
        usd_cost_estimate: Estimated direct monetary cost in USD.
    """

    token_cost_estimate: int | None = None
    latency_ms_estimate: int | None = None
    usd_cost_estimate: float | None = None


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Static description of a tool available to agent runtimes.

    Attributes:
        name: Stable tool identifier used for invocation.
        description: Natural-language description used for planning/routing.
        input_schema: JSON-schema-like object describing accepted inputs.
        output_schema: JSON-schema-like object describing tool outputs.
        permissions: Optional permission labels associated with the tool.
        cost_hints: Optional cost estimates used for planning heuristics.
    """

    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    permissions: tuple[str, ...] = ()
    cost_hints: ToolCostHints = field(default_factory=ToolCostHints)


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Result payload emitted from a tool runtime invocation.

    Attributes:
        tool_name: Name of the invoked tool.
        output: Structured tool output payload.
        success: Boolean execution success flag.
        error: Optional error text when execution failed.
        metadata: Optional runtime metadata (timings, trace IDs, etc.).
    """

    tool_name: str
    output: dict[str, object]
    success: bool
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class ToolRuntime(Protocol):
    """Protocol for registering and invoking named tools.

    Implementations may be in-memory, remote, or hybrid, but must present the
    same listing and invocation interface to agents.
    """

    def list_tools(self) -> Sequence[ToolSpec]:
        """Return all currently registered tool specifications.

        Returned specs describe every tool callable through ``invoke``.
        """

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        context: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one tool using structured input and execution context payloads.

        Implementations should avoid raising for expected tool failures and
        instead return ``ToolResult(success=False)`` with error details.
        """
