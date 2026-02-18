"""Unified execution result contract shared by agents and workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .llm import LLMResponse
from .tools import ToolResult


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    """Structured output produced by one execution entrypoint.

    This shape intentionally covers both agent-like executions and workflow-like
    executions so callers can consume one result contract everywhere.
    """

    success: bool
    """True when the overall run completed without terminal failure."""
    output: dict[str, object] = field(default_factory=dict)
    """Primary payload produced by the entrypoint."""
    tool_results: list[ToolResult] = field(default_factory=list)
    """Tool invocation results captured during execution, in call order."""
    model_response: LLMResponse | None = None
    """Final model response associated with the run, when available."""
    step_results: dict[str, Any] = field(default_factory=dict)
    """Per-step results keyed by step id for workflow-style runs."""
    execution_order: list[str] = field(default_factory=list)
    """Step ids in the order they were executed for workflow-style runs."""
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
        return f"ExecutionResult({self.asdict()!r})"
