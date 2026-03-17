"""Delegate runtime protocol shared by all concrete executable delegates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from ._execution import ExecutionResult

if TYPE_CHECKING:
    from design_research_agents.workflow._compiled import CompiledExecution


class Delegate(Protocol):
    """Protocol that every direct delegate implementation must satisfy.

    The protocol intentionally keeps the execution contract small: one
    compile phase plus one non-streaming execution call.
    """

    def compile(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile one delegate run into a bound workflow execution."""

    def compile_to_mermaid(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
        direction: str = "TD",
    ) -> str:
        """Compile one delegate run and return a Mermaid workflow diagram."""
        compiled = self.compile(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        return compiled.to_mermaid(direction=direction)

    def compile_to_svg(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
        direction: str = "TD",
    ) -> str:
        """Compile one delegate run and return an SVG workflow diagram."""
        compiled = self.compile(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        return compiled.to_svg(direction=direction)

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one delegate run and return the final ``ExecutionResult`` payload.

        Implementations should treat ``prompt`` as the prompt text for one run.
        Use ``request_id`` and ``dependencies`` for run metadata and upstream
        dependency payloads.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final execution result payload.
        """


__all__ = ["Delegate", "ExecutionResult"]
