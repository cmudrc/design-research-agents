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
LoopStepContinuePredicate: TypeAlias = Callable[[int, Mapping[str, object]], bool]
LoopStepStateReducer: TypeAlias = Callable[
    [Mapping[str, object], "WorkflowResult", int],
    Mapping[str, object],
]
LoopStepTerminationReason = Literal[
    "condition_stopped",
    "max_iterations_reached",
    "iteration_failed",
]


@dataclass(slots=True, frozen=True)
class ToolStep:
    """Workflow step that invokes one runtime tool."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    tool_name: str
    """Registered tool name to invoke through the tool runtime."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before this step can run."""
    input_data: Mapping[str, object] | None = None
    """Static input payload used when ``input_builder`` is not provided."""
    input_builder: ToolStepInputBuilder | None = None
    """Optional callback that derives input payload from runtime step context."""


@dataclass(slots=True, frozen=True)
class AgentStep:
    """Workflow step that invokes one registered agent-like delegate."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    agent_name: str
    """Registered agent/delegate name to invoke for this step."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before this step can run."""
    prompt: str | None = None
    """Static prompt passed to the delegate when ``prompt_builder`` is absent."""
    prompt_builder: AgentStepPromptBuilder | None = None
    """Optional callback that derives a prompt string from runtime step context."""


@dataclass(slots=True, frozen=True)
class LogicStep:
    """Workflow step that executes deterministic local logic."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    handler: LogicStepHandler
    """Deterministic local function that computes this step output."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before this step can run."""
    route_map: Mapping[str, tuple[str, ...]] | None = None
    """Optional route key to downstream-target mapping for conditional activation."""


@dataclass(slots=True, frozen=True)
class LoopStep:
    """Workflow step that executes an iterative nested workflow body."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    steps: tuple[WorkflowStep, ...]
    """Static loop body steps executed for each iteration."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before loop iteration begins."""
    max_iterations: int = 1
    """Hard cap on the number of loop iterations."""
    initial_state: Mapping[str, object] | None = None
    """Initial loop state mapping provided to iteration context."""
    continue_predicate: LoopStepContinuePredicate | None = None
    """Predicate deciding whether to execute the next iteration."""
    state_reducer: LoopStepStateReducer | None = None
    """Reducer that computes next loop state from prior state and iteration result."""
    execution_mode: WorkflowExecutionMode = "sequential"
    """Execution mode used for nested loop-body workflow runs."""
    failure_policy: WorkflowFailurePolicy = "skip_dependents"
    """Failure handling policy applied within each loop iteration run."""


WorkflowStep: TypeAlias = ToolStep | AgentStep | LogicStep | LoopStep


@dataclass(slots=True, frozen=True)
class WorkflowStepResult:
    """Result payload for one workflow step execution."""

    step_id: str
    """Step id this result belongs to."""
    status: WorkflowStepStatus
    """Execution status for the step."""
    success: bool
    """True when step completed successfully."""
    output: dict[str, object] = field(default_factory=dict)
    """Step output payload produced by the runtime."""
    error: str | None = None
    """Human-readable error message when step fails."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Supplemental runtime metadata for diagnostics or tracing."""

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Dictionary representation of this step result.
        """
        return asdict(self)


@dataclass(slots=True, frozen=True)
class WorkflowResult:
    """Top-level result payload for one workflow run."""

    success: bool
    """True when the overall workflow run is successful."""
    step_results: dict[str, WorkflowStepResult]
    """Per-step results keyed by step id."""
    execution_order: list[str]
    """Step ids in the order they were executed."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Workflow-level metadata for diagnostics and orchestration context."""

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Dictionary representation of this workflow result.
        """
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
        """Execute a workflow definition and return aggregated results.

        Args:
            steps: Workflow step sequence to execute.
            context: Optional shared context mapping available to step builders.
            execution_mode: Global runtime scheduling mode (for example ``dag``).
            failure_policy: Global failure behavior when a step fails.
            request_id: Optional request id used for tracing and downstream calls.
            dependencies: Optional dependency payload mapping exposed to steps.

        Returns:
            Aggregated workflow execution result.
        """


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
        """Execute the configured orchestration and return aggregated results.

        Args:
            context: Optional shared context mapping available to step builders.
            execution_mode: Runtime scheduling mode (for example ``dag``).
            failure_policy: Failure behavior when a step fails.
            request_id: Optional request id used for tracing and downstream calls.
            dependencies: Optional dependency payload mapping exposed to steps.

        Returns:
            Aggregated workflow execution result.
        """


WorkflowDelegate: TypeAlias = Agent | WorkflowDelegateRunner
