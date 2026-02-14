"""Tool specification and runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True, frozen=True)
class ToolCostHints:
    """Approximate cost metadata associated with a tool invocation."""

    token_cost_estimate: int | None = None
    latency_ms_estimate: int | None = None
    usd_cost_estimate: float | None = None


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Static description of a tool."""

    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    permissions: tuple[str, ...] = ()
    cost_hints: ToolCostHints = field(default_factory=ToolCostHints)


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Result emitted from a tool runtime invocation."""

    tool_name: str
    output: dict[str, object]
    success: bool
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class ToolRuntime(Protocol):
    """Protocol for invoking named tools."""

    def list_tools(self) -> Sequence[ToolSpec]:
        """Return all registered tool specifications."""

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        context: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one tool with structured input and execution context."""
