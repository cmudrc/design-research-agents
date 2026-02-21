"""RAG-first reasoning pattern built from workflow memory primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.memory import MemoryStore
from design_research_agents.contracts.workflow import (
    AgentStep,
    MemoryReadStep,
    MemoryWriteStep,
    WorkflowDelegate,
    WorkflowStep,
)
from design_research_agents.implementations.shared.workflow_internal import (
    execute_pattern_with_trace,
    resolve_pattern_run_context,
)
from design_research_agents.tracing import Tracer
from design_research_agents.workflow.workflow import Workflow


class RagReasoningPattern(Agent):
    """Reasoning pattern orchestrated as memory read -> reason -> memory write."""

    def __init__(
        self,
        *,
        reasoning_delegate: WorkflowDelegate,
        memory_store: MemoryStore | None,
        memory_namespace: str = "default",
        memory_top_k: int = 5,
        memory_min_score: float | None = None,
        write_back: bool = True,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize RAG reasoning pattern.

        Args:
            reasoning_delegate: Delegate object that performs reasoning with
                retrieved context.
            memory_store: Memory store used for retrieval and optional write-back.
            memory_namespace: Namespace partition for reads/writes.
            memory_top_k: Number of retrieved matches for reasoning context.
            memory_min_score: Optional minimum retrieval score threshold.
            write_back: Whether to persist one summary record after reasoning.
            tracer: Optional tracer dependency.

        Raises:
            ValueError: Raised when ``memory_top_k`` is less than one.
        """
        if memory_top_k < 1:
            raise ValueError("memory_top_k must be >= 1.")

        self._reasoning_delegate = reasoning_delegate
        self._memory_store = memory_store
        self._memory_namespace = memory_namespace.strip() or "default"
        self._memory_top_k = memory_top_k
        self._memory_min_score = memory_min_score
        self._write_back = write_back
        self._tracer = tracer
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute memory retrieval, delegated reasoning, and optional write-back.

        Args:
            prompt: Task prompt.
            request_id: Optional request identifier.
            dependencies: Optional dependency mapping.

        Returns:
            Aggregated RAG reasoning result.

        Raises:
            Exception: Propagated workflow runtime errors.
        """
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=None,
            default_dependencies={},
            request_id=request_id,
            dependencies=dependencies,
        )
        return execute_pattern_with_trace(
            agent_name="RagReasoningPattern",
            request_id=run_context.request_id,
            input_payload={
                "prompt": prompt,
                "memory_namespace": self._memory_namespace,
                "memory_top_k": self._memory_top_k,
                "write_back": self._write_back,
            },
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            runner=lambda: self._run_rag_reasoning(
                prompt=prompt,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def _run_rag_reasoning(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Execute underlying workflow for read/reason/write orchestration.

        Args:
            prompt: Task prompt.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency mapping.

        Returns:
            Aggregated workflow result.
        """
        workflow_steps: list[WorkflowStep] = [
            MemoryReadStep(
                step_id="memory_read",
                query_builder=lambda context: str(context.get("prompt", "")),
                namespace=self._memory_namespace,
                top_k=self._memory_top_k,
                min_score=self._memory_min_score,
            ),
            AgentStep(
                step_id="reason",
                dependencies=("memory_read",),
                delegate=self._reasoning_delegate,
                prompt_builder=lambda context: _build_reasoning_prompt(
                    task_prompt=str(context.get("prompt", "")),
                    memory_read_step_output=_extract_dependency_output(
                        context=context,
                        dependency_id="memory_read",
                    ),
                ),
            ),
        ]
        if self._write_back:
            workflow_steps.append(
                MemoryWriteStep(
                    step_id="memory_write",
                    dependencies=("memory_read", "reason"),
                    namespace=self._memory_namespace,
                    records_builder=lambda context: _build_write_back_records(
                        task_prompt=str(context.get("prompt", "")),
                        reason_step_output=_extract_dependency_output(
                            context=context,
                            dependency_id="reason",
                        ),
                        memory_read_step_output=_extract_dependency_output(
                            context=context,
                            dependency_id="memory_read",
                        ),
                    ),
                )
            )

        self.workflow = Workflow(
            tool_runtime=None,
            memory_store=self._memory_store,
            tracer=self._tracer,
            input_mode="schema",
            base_context={"prompt": prompt},
            steps=workflow_steps,
        )

        workflow_result = self.workflow.run(
            {},
            execution_mode="sequential",
            request_id=f"{request_id}:rag_reasoning",
            dependencies=dependencies,
        )

        memory_read_result = workflow_result.step_results.get("memory_read")
        reason_result = workflow_result.step_results.get("reason")
        memory_write_result = workflow_result.step_results.get("memory_write")

        retrieval_output = (
            dict(memory_read_result.output)
            if memory_read_result is not None
            else {
                "query": {},
                "matches": [],
                "count": 0,
                "namespace": self._memory_namespace,
            }
        )
        reasoning_output = dict(reason_result.output) if reason_result is not None else {}
        write_back_output = (
            dict(memory_write_result.output)
            if memory_write_result is not None
            else {"written": 0, "namespace": self._memory_namespace, "ids": []}
        )
        workflow_payload = workflow_result.asdict()
        workflow_artifacts = workflow_result.output.get("artifacts", [])
        delegate_final_output = reasoning_output.get("output")
        final_output = (
            dict(delegate_final_output)
            if isinstance(delegate_final_output, Mapping)
            else dict(reasoning_output)
        )

        result_success = workflow_result.success
        terminated_reason = "completed" if result_success else "workflow_failure"

        return ExecutionResult(
            output={
                "retrieval": retrieval_output,
                "reasoning": reasoning_output,
                "write_back": write_back_output,
                "workflow": workflow_payload,
                "final_output": final_output,
                "artifacts": workflow_artifacts,
                "terminated_reason": terminated_reason,
            },
            success=result_success,
            tool_results=[],
            model_response=None,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "memory_namespace": self._memory_namespace,
                "memory_top_k": self._memory_top_k,
                "write_back": self._write_back,
            },
        )


def _extract_dependency_output(
    *,
    context: Mapping[str, object],
    dependency_id: str,
) -> Mapping[str, object]:
    """Extract one dependency output mapping from workflow step context.

    Args:
        context: Step context mapping.
        dependency_id: Dependency step identifier.

    Returns:
        Dependency output mapping when present, otherwise empty mapping.
    """
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return {}
    dependency_payload = dependency_results.get(dependency_id)
    if not isinstance(dependency_payload, Mapping):
        return {}
    output = dependency_payload.get("output")
    if isinstance(output, Mapping):
        return output
    return {}


def _build_reasoning_prompt(
    *,
    task_prompt: str,
    memory_read_step_output: Mapping[str, object],
) -> str:
    """Build explicit prompt with retrieved context injection.

    Args:
        task_prompt: Task prompt.
        memory_read_step_output: Output payload from memory read step.

    Returns:
        Prompt string passed to the reasoning delegate.
    """
    matches = memory_read_step_output.get("matches")
    normalized_matches = matches if isinstance(matches, list) else []

    context_json_block = json.dumps(
        {
            "namespace": memory_read_step_output.get("namespace", "default"),
            "count": _safe_int(memory_read_step_output.get("count")),
            "matches": normalized_matches,
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )

    context_text_lines: list[str] = []
    for match in normalized_matches:
        if not isinstance(match, Mapping):
            continue
        item_id = str(match.get("item_id", ""))
        score = match.get("score")
        content = str(match.get("content", "")).strip()
        if not content:
            continue
        score_text = f" score={score}" if isinstance(score, (int, float)) else ""
        context_text_lines.append(f"- [{item_id}]{score_text} {content}")
    context_text = "\n".join(context_text_lines) if context_text_lines else "(none)"

    prompt_lines = [
        f"Task: {task_prompt}",
        "",
        "Retrieved context (JSON):",
        context_json_block,
        "",
        "Retrieved context (text):",
        context_text,
        "",
        "Use the retrieved context when relevant, but reason independently when context is sparse.",
    ]
    return "\n".join(prompt_lines)


def _build_write_back_records(
    *,
    task_prompt: str,
    reason_step_output: Mapping[str, object],
    memory_read_step_output: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build write-back records from reasoning output and retrieval context.

    Args:
        task_prompt: Task prompt.
        reason_step_output: Reasoning step output payload.
        memory_read_step_output: Memory read step output payload.

    Returns:
        Memory write payloads for optional persistence.
    """
    reasoning_payload = reason_step_output.get("output")
    normalized_reasoning = dict(reasoning_payload) if isinstance(reasoning_payload, Mapping) else {}

    retrieval_matches = memory_read_step_output.get("matches")
    retrieved_count = len(retrieval_matches) if isinstance(retrieval_matches, list) else 0

    content_payload = {
        "task": task_prompt,
        "retrieved_count": retrieved_count,
        "reasoning": normalized_reasoning,
    }
    return [
        {
            "content": json.dumps(content_payload, ensure_ascii=True, sort_keys=True),
            "metadata": {
                "kind": "rag_reasoning",
                "retrieved_count": retrieved_count,
                "task": task_prompt,
            },
        }
    ]


def _safe_int(value: object) -> int:
    """Convert values to int with deterministic fallback to zero.

    Args:
        value: Raw input value.

    Returns:
        Integer representation or ``0`` fallback.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


__all__ = ["RagReasoningPattern"]
