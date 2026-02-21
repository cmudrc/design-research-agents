"""Workflow runtime contracts and typed step payloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from .agent import Agent
from .execution import ExecutionResult
from .llm import LLMClient, LLMRequest, LLMResponse
from .memory import MemoryWriteRecord

WorkflowExecutionMode = Literal["sequential", "dag"]
WorkflowFailurePolicy = Literal["skip_dependents", "propagate_failed_state"]
WorkflowStepStatus = Literal["completed", "failed", "skipped"]


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
    ) -> ExecutionResult:
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


@runtime_checkable
class WorkflowObjectDelegate(Protocol):
    """Protocol for raw ``Workflow`` objects used as delegates."""

    def run(
        self,
        input_data: str | Mapping[str, object] | None = None,
        *,
        execution_mode: WorkflowExecutionMode = "sequential",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute a workflow object and return one aggregate result.

        Args:
            input_data: Optional workflow input payload.
            execution_mode: Runtime scheduling mode (for example ``dag``).
            failure_policy: Failure behavior when a step fails.
            request_id: Optional request id used for tracing and downstream calls.
            dependencies: Optional dependency payload mapping exposed to steps.

        Returns:
            Aggregated workflow execution result.
        """


WorkflowDelegate: TypeAlias = Agent | WorkflowDelegateRunner | WorkflowObjectDelegate


@dataclass(slots=True, frozen=True)
class WorkflowArtifactSource:
    """Provenance entry describing one artifact source edge."""

    step_id: str
    """Step id that contributed to this artifact."""
    field: str | None = None
    """Optional output field or source label within the step payload."""
    note: str | None = None
    """Optional human-readable provenance note."""


@dataclass(slots=True, frozen=True)
class WorkflowArtifact:
    """User-facing workflow artifact manifest entry."""

    path: str
    """Filesystem path to the artifact."""
    mime: str
    """MIME type for artifact consumers."""
    title: str | None = None
    """Optional short artifact title for UIs."""
    summary: str | None = None
    """Optional artifact summary for user-facing rendering."""
    audience: str | None = None
    """Optional target audience label (for example ``user``)."""
    producer_step_id: str | None = None
    """Step id that produced the artifact when known."""
    sources: tuple[WorkflowArtifactSource, ...] = ()
    """Provenance entries describing source steps/fields."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Supplemental artifact metadata."""

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Dictionary representation of this artifact entry.
        """
        return asdict(self)


WorkflowArtifactsBuilder: TypeAlias = Callable[
    [Mapping[str, object]],
    Sequence[WorkflowArtifact | Mapping[str, object]],
]
ToolStepInputBuilder: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, object]]
AgentStepPromptBuilder: TypeAlias = Callable[[Mapping[str, object]], str]
ModelStepRequestBuilder: TypeAlias = Callable[[Mapping[str, object]], LLMRequest]
ModelStepResponseParser: TypeAlias = Callable[
    [LLMResponse, Mapping[str, object]],
    Mapping[str, object],
]
LogicStepHandler: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, object]]
MemoryReadQueryBuilder: TypeAlias = Callable[[Mapping[str, object]], str | Mapping[str, object]]
MemoryWriteRecordsBuilder: TypeAlias = Callable[
    [Mapping[str, object]],
    Sequence[str | Mapping[str, object] | MemoryWriteRecord],
]
LoopStepContinuePredicate: TypeAlias = Callable[[int, Mapping[str, object]], bool]
LoopStepStateReducer: TypeAlias = Callable[
    [Mapping[str, object], ExecutionResult, int],
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
    artifacts_builder: WorkflowArtifactsBuilder | None = None
    """Optional callback that extracts user-facing artifact manifests from step context."""


@dataclass(slots=True, frozen=True)
class AgentStep:
    """Workflow step that invokes one direct delegate."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    delegate: WorkflowDelegate
    """Direct delegate object (agent, pattern, or workflow-like runner)."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before this step can run."""
    prompt: str | None = None
    """Static prompt passed to the delegate when ``prompt_builder`` is absent."""
    prompt_builder: AgentStepPromptBuilder | None = None
    """Optional callback that derives a prompt string from runtime step context."""
    artifacts_builder: WorkflowArtifactsBuilder | None = None
    """Optional callback that extracts user-facing artifact manifests from step context."""


@dataclass(slots=True, frozen=True)
class ModelStep:
    """Workflow step that executes one model request through an LLM client."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    llm_client: LLMClient
    """LLM client used to execute the request built for this step."""
    request_builder: ModelStepRequestBuilder
    """Callback that builds the ``LLMRequest`` payload from runtime context."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before this step can run."""
    response_parser: ModelStepResponseParser | None = None
    """Optional callback that parses model response into structured output."""
    artifacts_builder: WorkflowArtifactsBuilder | None = None
    """Optional callback that extracts user-facing artifact manifests from step context."""


@dataclass(slots=True, frozen=True)
class DelegateBatchCall:
    """One delegate call specification executed by ``DelegateBatchStep``."""

    call_id: str
    """Unique call identifier within the batch."""
    delegate: WorkflowDelegate
    """Delegate object invoked for this call."""
    prompt: str
    """Prompt passed to the delegate for this call."""
    execution_mode: WorkflowExecutionMode = "sequential"
    """Execution mode propagated when the delegate is workflow-like."""
    failure_policy: WorkflowFailurePolicy = "skip_dependents"
    """Failure policy propagated when the delegate is workflow-like."""


DelegateBatchCallsBuilder: TypeAlias = Callable[
    [Mapping[str, object]],
    Sequence[DelegateBatchCall | Mapping[str, object]],
]


@dataclass(slots=True, frozen=True)
class DelegateBatchStep:
    """Workflow step that executes multiple delegate invocations in sequence."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    calls_builder: DelegateBatchCallsBuilder
    """Callback that builds batch delegate call specs from runtime context."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before this step can run."""
    fail_fast: bool = True
    """Whether to stop executing additional calls after first failure."""
    artifacts_builder: WorkflowArtifactsBuilder | None = None
    """Optional callback that extracts user-facing artifact manifests from step context."""


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
    artifacts_builder: WorkflowArtifactsBuilder | None = None
    """Optional callback that extracts user-facing artifact manifests from step context."""


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
    artifacts_builder: WorkflowArtifactsBuilder | None = None
    """Optional callback that extracts user-facing artifact manifests from step context."""


@dataclass(slots=True, frozen=True)
class MemoryReadStep:
    """Workflow step that reads relevant records from the memory store."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    query_builder: MemoryReadQueryBuilder
    """Callback that builds query text or query payload from step context."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before this step can run."""
    namespace: str = "default"
    """Namespace partition to read from."""
    top_k: int = 5
    """Maximum number of records to return."""
    min_score: float | None = None
    """Optional minimum score threshold for returned records."""
    artifacts_builder: WorkflowArtifactsBuilder | None = None
    """Optional callback that extracts user-facing artifact manifests from step context."""


@dataclass(slots=True, frozen=True)
class MemoryWriteStep:
    """Workflow step that writes records into the memory store."""

    step_id: str
    """Unique step identifier used for dependency wiring and result lookup."""
    records_builder: MemoryWriteRecordsBuilder
    """Callback that builds record payloads from step context."""
    dependencies: tuple[str, ...] = ()
    """Step ids that must complete before this step can run."""
    namespace: str = "default"
    """Namespace partition to write into."""
    artifacts_builder: WorkflowArtifactsBuilder | None = None
    """Optional callback that extracts user-facing artifact manifests from step context."""


WorkflowStep: TypeAlias = (
    ToolStep
    | AgentStep
    | ModelStep
    | DelegateBatchStep
    | LogicStep
    | LoopStep
    | MemoryReadStep
    | MemoryWriteStep
)


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
    artifacts: tuple[WorkflowArtifact, ...] = ()
    """User-facing artifact manifests produced by this step."""

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Dictionary representation of this step result.
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
    ) -> ExecutionResult:
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
