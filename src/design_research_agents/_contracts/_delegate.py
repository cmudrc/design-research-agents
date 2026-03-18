"""Delegate runtime protocol shared by all concrete executable delegates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from ._execution import ExecutionResult

if TYPE_CHECKING:
    from design_research_agents.workflow import Workflow
    from design_research_agents.workflow._compiled import CompiledExecution


def _require_compiled_workflow(delegate: object) -> Workflow:
    """Return the most recently compiled workflow attached to a delegate."""
    from design_research_agents.workflow import Workflow

    workflow = getattr(delegate, "workflow", None)
    if isinstance(workflow, Workflow):
        return workflow
    raise RuntimeError(
        "No compiled workflow is available on this delegate. "
        "Compile the delegate once before calling compile_to_mermaid() or compile_to_svg(), "
        "or render from the returned CompiledExecution instead."
    )


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
        *,
        direction: str = "TD",
    ) -> str:
        """Render Mermaid for the most recently compiled workflow on this delegate."""
        workflow = _require_compiled_workflow(self)
        return workflow.to_mermaid(direction=direction)

    def compile_to_svg(
        self,
        *,
        direction: str = "TD",
    ) -> str:
        """Render SVG for the most recently compiled workflow on this delegate."""
        workflow = _require_compiled_workflow(self)
        return workflow.to_svg(direction=direction)

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
