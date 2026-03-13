"""Reusable configurable workflow facade for user-supplied step graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._memory import MemoryStore
from design_research_agents._contracts._tools import ToolRuntime
from design_research_agents._contracts._workflow import (
    ExecutionResult,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowStep,
)
from design_research_agents._runtime._common._run_defaults import (
    merge_dependencies,
    normalize_request_id_prefix,
    resolve_request_id_with_prefix,
)
from design_research_agents._runtime._workflow._engine import WorkflowRuntime
from design_research_agents._schemas import validate_payload_against_schema
from design_research_agents._tracing import Tracer

from ._diagram import render_workflow_as_mermaid


def _normalize_steps(steps: Sequence[WorkflowStep]) -> tuple[WorkflowStep, ...]:
    """Validate and freeze configured workflow steps.

    Args:
        steps: Workflow steps supplied to the facade constructor.

    Returns:
        Immutable step tuple used for runtime execution.

    Raises:
        ValueError: If no steps were provided.
    """
    if not steps:
        raise ValueError("'steps' must contain at least one workflow step.")
    return tuple(steps)


def _normalize_prompt(input_value: object) -> str:
    """Validate and normalize prompt-mode workflow input.

    Args:
        input_value: Raw workflow input value.

    Returns:
        Trimmed prompt string.

    Raises:
        ValueError: If input is not a non-empty string.
    """
    if not isinstance(input_value, str):
        raise ValueError("Workflow configured without input_schema requires string input.")
    normalized_prompt = input_value.strip()
    if not normalized_prompt:
        raise ValueError("Workflow prompt input must be a non-empty string.")
    return normalized_prompt


def _normalize_inputs(input_value: object) -> dict[str, object]:
    """Validate and normalize schema-mode workflow input.

    Args:
        input_value: Raw workflow input value.

    Returns:
        Input mapping copied into a mutable dictionary.

    Raises:
        ValueError: If input is not a mapping when provided.
    """
    if input_value is None:
        return {}
    if not isinstance(input_value, Mapping):
        raise ValueError("Workflow configured with input_schema requires mapping input.")
    return dict(input_value)


class Workflow:
    """Configured workflow for user-defined step graphs and run defaults."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime | None = None,
        memory_store: MemoryStore | None = None,
        steps: Sequence[WorkflowStep],
        input_schema: Mapping[str, object] | None = None,
        output_schema: Mapping[str, object] | None = None,
        prompt_context_key: str = "prompt",
        base_context: Mapping[str, object] | None = None,
        default_execution_mode: WorkflowExecutionMode = "sequential",
        default_failure_policy: WorkflowFailurePolicy = "skip_dependents",
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store runtime dependencies, step graph, and input handling mode.

        Args:
            tool_runtime: Tool runtime used by ``ToolStep`` executions.
            memory_store: Optional memory store used by memory step executions.
            steps: Static workflow step graph to execute for each run.
            input_schema: Optional schema used to infer input mode and validate mapped input.
                When omitted, workflow expects prompt-string input.
            output_schema: Optional schema enforced against ``output.final_output`` when the run
                succeeds.
            prompt_context_key: Context key used to store normalized prompt input.
            base_context: Base context merged into every run context.
            default_execution_mode: Default runtime step scheduling mode.
            default_failure_policy: Default dependency failure handling policy.
            default_request_id_prefix: Optional prefix used to derive request ids.
            default_dependencies: Default dependency objects injected into each run.
            tracer: Optional tracer used for workflow runtime events.

        Raises:
            ValueError: If constructor inputs are inconsistent.
        """
        normalized_prompt_context_key = prompt_context_key.strip()
        if input_schema is None and not normalized_prompt_context_key:
            raise ValueError("prompt_context_key must be non-empty when input_schema is omitted.")

        self._runtime = WorkflowRuntime(
            tool_runtime=tool_runtime,
            memory_store=memory_store,
            tracer=tracer,
        )
        self._steps = _normalize_steps(steps)
        self._input_schema = dict(input_schema) if input_schema is not None else None
        self._output_schema = dict(output_schema) if output_schema is not None else None
        self._prompt_context_key = normalized_prompt_context_key or "prompt"
        self._base_context = dict(base_context or {})
        self._default_execution_mode = default_execution_mode
        self._default_failure_policy = default_failure_policy
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})

    def to_mermaid(self, *, direction: str = "TD") -> str:
        """Return a deterministic Mermaid diagram for the configured workflow.

        Args:
            direction: Mermaid flowchart direction (for example ``TD`` or ``LR``).

        Returns:
            Mermaid flowchart text that reflects the declared step topology.
        """
        return render_workflow_as_mermaid(self._steps, direction=direction)

    def run(
        self,
        input: str | Mapping[str, object] | None = None,
        *,
        execution_mode: WorkflowExecutionMode | None = None,
        failure_policy: WorkflowFailurePolicy | None = None,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one workflow run with input mode inferred from ``input_schema``.

        Args:
            input: Prompt string when ``input_schema`` is omitted; otherwise schema mapping.
            execution_mode: Optional per-run execution mode override.
            failure_policy: Optional per-run failure policy override.
            request_id: Optional explicit request id for tracing/correlation.
            dependencies: Optional per-run dependency overrides.

        Returns:
            Aggregated workflow execution result.
        """
        resolved_request_id = resolve_request_id_with_prefix(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        context = dict(self._base_context)
        if self._input_schema is None:
            normalized_prompt = _normalize_prompt(input)
            context[self._prompt_context_key] = normalized_prompt
        else:
            normalized_inputs = _normalize_inputs(input)
            validate_payload_against_schema(
                payload=normalized_inputs,
                schema=self._input_schema,
                location="inputs",
            )
            context["inputs"] = normalized_inputs
        if self._output_schema is not None:
            context["_workflow_output_schema"] = dict(self._output_schema)

        workflow_result = self._runtime.run(
            self._steps,
            context=context,
            execution_mode=execution_mode or self._default_execution_mode,
            failure_policy=failure_policy or self._default_failure_policy,
            request_id=resolved_request_id,
            dependencies=merge_dependencies(
                default_dependencies=self._default_dependencies,
                run_dependencies=dependencies,
            ),
        )
        if workflow_result.success:
            validate_payload_against_schema(
                payload=workflow_result.output.get("final_output"),
                schema=self._output_schema,
                location="output.final_output",
            )
        return workflow_result


__all__ = [
    "Workflow",
]
