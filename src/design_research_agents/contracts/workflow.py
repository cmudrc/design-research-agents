"""Workflow runtime contracts and typed step payloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from .agent import Agent

WorkflowExecutionMode = Literal["sequential", "dag"]
WorkflowFailurePolicy = Literal["skip_dependents", "propagate_failed_state"]
WorkflowStepStatus = Literal["completed", "failed", "skipped"]

ToolStepInputBuilder: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, object]]
AgentStepPromptBuilder: TypeAlias = Callable[[Mapping[str, object]], str]
LogicStepHandler: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(slots=True, frozen=True)
class ToolStep:
    """Workflow step that invokes one runtime tool."""

    step_id: str
    tool_name: str
    dependencies: tuple[str, ...] = ()
    input_data: Mapping[str, object] | None = None
    input_builder: ToolStepInputBuilder | None = None


@dataclass(slots=True, frozen=True)
class AgentStep:
    """Workflow step that invokes one registered agent-like delegate."""

    step_id: str
    agent_name: str
    dependencies: tuple[str, ...] = ()
    prompt: str | None = None
    prompt_builder: AgentStepPromptBuilder | None = None


@dataclass(slots=True, frozen=True)
class LogicStep:
    """Workflow step that executes deterministic local logic."""

    step_id: str
    handler: LogicStepHandler
    dependencies: tuple[str, ...] = ()
    route_map: Mapping[str, tuple[str, ...]] | None = None


WorkflowStep: TypeAlias = ToolStep | AgentStep | LogicStep


@dataclass(slots=True, frozen=True)
class WorkflowStepResult:
    """Result payload for one workflow step execution."""

    step_id: str
    status: WorkflowStepStatus
    success: bool
    output: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


@dataclass(slots=True, frozen=True)
class WorkflowResult:
    """Top-level result payload for one workflow run."""

    success: bool
    step_results: dict[str, WorkflowStepResult]
    execution_order: list[str]
    metadata: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


class WorkflowRunner(Protocol):
    """Protocol implemented by workflow runtime implementations."""

    def run(
        self,
        steps: Sequence[WorkflowStep],
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode = "dag",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute a workflow definition and return aggregated results."""


@runtime_checkable
class WorkflowDelegateRunner(Protocol):
    """Protocol for configured orchestration chunks with fixed step topology."""

    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode = "dag",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute the configured orchestration and return aggregated results."""


WorkflowDelegate: TypeAlias = Agent | WorkflowDelegateRunner
