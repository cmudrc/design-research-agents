"""Workflow orchestration contracts and result payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

WorkflowFailurePolicy = Literal["skip_dependents", "propagate_failed_state"]
WorkflowNodeStatus = Literal["completed", "failed", "skipped"]


@dataclass(slots=True, frozen=True)
class WorkflowNodeResult:
    """Result payload for one workflow node execution."""

    node_id: str
    status: WorkflowNodeStatus
    success: bool
    output: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        return asdict(self)


@dataclass(slots=True, frozen=True)
class WorkflowResult:
    """Top-level result payload for one workflow run."""

    success: bool
    node_results: dict[str, WorkflowNodeResult]
    execution_order: list[str]
    metadata: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        return asdict(self)


class WorkflowNode(Protocol):
    """Contract implemented by workflow nodes."""

    node_id: str
    dependencies: Sequence[str]
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]

    def run(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Execute node logic and return schema-conforming output."""


class Orchestrator(Protocol):
    """Protocol implemented by workflow orchestrator runtimes."""

    def run(
        self,
        nodes: Sequence[WorkflowNode],
        *,
        context: Mapping[str, object] | None = None,
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute a workflow definition."""
