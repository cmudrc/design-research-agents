"""Unified execution result contract shared by agents and workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ._llm import LLMResponse
from ._tools import ToolResult


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the result.

        Returns:
            Dictionary representation of the result payload.
        """
        return asdict(self)

    @property
    def final_output(self) -> object | None:
        """Return workflow/agent ``final_output`` payload when present.

        Returns:
            Final output value from ``output`` payload, or ``None``.
        """
        return self.output.get("final_output")

    @property
    def terminated_reason(self) -> str | None:
        """Return normalized termination reason when present.

        Returns:
            Termination reason string, or ``None``.
        """
        value = self.output.get("terminated_reason")
        return value if isinstance(value, str) else None

    @property
    def error(self) -> object | None:
        """Return terminal error payload when present.

        Returns:
            Error payload from ``output`` mapping, or ``None``.
        """
        return self.output.get("error")

    def output_value(self, key: str, default: object | None = None) -> object | None:
        """Return one output value by key with optional default.

        Args:
            key: Output key to read.
            default: Value returned when ``key`` is absent.

        Returns:
            Output value for ``key`` when present, else ``default``.
        """
        return self.output.get(key, default)

    def output_dict(self, key: str) -> dict[str, object]:
        """Return one output value normalized to a dictionary.

        Args:
            key: Output key to read.

        Returns:
            Dictionary value when the output value is mapping-like, else ``{}``.
        """
        value = self.output.get(key)
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def output_list(self, key: str) -> list[object]:
        """Return one output value normalized to a list.

        Args:
            key: Output key to read.

        Returns:
            List value when the output value is a list/tuple, else ``[]``.
        """
        value = self.output.get(key)
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        return []

    def to_json(
        self,
        *,
        ensure_ascii: bool = True,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str:
        """Return JSON string for deterministic pretty-printing.

        Args:
            ensure_ascii: Forwarded to ``json.dumps``.
            indent: Forwarded to ``json.dumps``.
            sort_keys: Forwarded to ``json.dumps``.

        Returns:
            JSON representation of this result.
        """
        return json.dumps(
            self.to_dict(),
            ensure_ascii=ensure_ascii,
            indent=indent,
            sort_keys=sort_keys,
        )

    def __str__(self) -> str:
        """Return a JSON-formatted string representation of the result.

        Returns:
            Pretty-printed JSON string for the result.
        """
        return self.to_json()

    def __repr__(self) -> str:
        """Return a human-readable string representation of the result.

        Returns:
            Debug-oriented string representation.
        """
        return f"ExecutionResult({self.to_dict()!r})"
