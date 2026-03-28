"""RAG-first reasoning pattern built from workflow memory primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._memory import GraphMemoryStore, GraphSearchQuery, MemoryStore
from design_research_agents._contracts._workflow import (
    DelegateStep,
    DelegateTarget,
    LogicStep,
    MemoryReadStep,
    MemoryWriteStep,
    WorkflowStep,
)
from design_research_agents._runtime._patterns import (
    MODE_RAG,
    build_compiled_pattern_execution,
    build_pattern_execution_result,
    resolve_pattern_run_context,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution
from design_research_agents.workflow.workflow import Workflow


class RAGPattern(Delegate):
    """Reasoning pattern orchestrated as memory read -> reason -> memory write."""

    def __init__(
        self,
        *,
        reasoning_delegate: DelegateTarget,
        memory_store: MemoryStore | None,
        memory_namespace: str = "default",
        memory_top_k: int = 5,
        memory_min_score: float | None = None,
        graph_memory_store: GraphMemoryStore | None = None,
        graph_namespace: str | None = None,
        graph_top_k: int = 3,
        graph_max_hops: int = 1,
        graph_min_score: float | None = None,
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
            graph_memory_store: Optional graph memory store used for subgraph retrieval.
            graph_namespace: Optional namespace for graph retrieval; defaults to ``memory_namespace``.
            graph_top_k: Number of graph seed nodes to retrieve.
            graph_max_hops: Graph traversal depth from matched seed nodes.
            graph_min_score: Optional minimum graph seed score threshold.
            write_back: Whether to persist one summary record after reasoning.
            tracer: Optional tracer dependency.

        Raises:
            ValueError: Raised when retrieval configuration is invalid.
        """
        if memory_top_k < 1:
            raise ValueError("memory_top_k must be >= 1.")
        if graph_top_k < 1:
            raise ValueError("graph_top_k must be >= 1.")
        if graph_max_hops < 0:
            raise ValueError("graph_max_hops must be >= 0.")

        self._reasoning_delegate = reasoning_delegate
        self._memory_store = memory_store
        self._memory_namespace = memory_namespace.strip() or "default"
        self._memory_top_k = memory_top_k
        self._memory_min_score = memory_min_score
        self._graph_memory_store = graph_memory_store
        self._graph_namespace = (graph_namespace or self._memory_namespace).strip() or self._memory_namespace
        self._graph_top_k = graph_top_k
        self._graph_max_hops = graph_max_hops
        self._graph_min_score = graph_min_score
        self._write_back = write_back
        self._tracer = tracer
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute memory retrieval, delegated reasoning, and optional write-back."""
        return self.compile(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def compile(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile the read/reason/write workflow."""
        run_context = resolve_pattern_run_context(
            prompt=prompt,
            default_request_id_prefix=None,
            default_dependencies={},
            request_id=request_id,
            dependencies=dependencies,
        )
        input_payload = {
            **run_context.normalized_input,
            "mode": MODE_RAG,
            "memory_namespace": self._memory_namespace,
            "memory_top_k": self._memory_top_k,
            "graph_namespace": self._graph_namespace,
            "graph_top_k": self._graph_top_k,
            "graph_max_hops": self._graph_max_hops,
            "write_back": self._write_back,
        }
        workflow = self._build_workflow(
            run_context.prompt,
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
        )
        return build_compiled_pattern_execution(
            workflow=workflow,
            pattern_name="RAGPattern",
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            input_payload=input_payload,
            workflow_request_id=f"{run_context.request_id}:rag_reasoning",
            finalize=lambda workflow_result: _build_rag_result(
                workflow_result=workflow_result,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
                memory_namespace=self._memory_namespace,
                memory_top_k=self._memory_top_k,
                graph_namespace=self._graph_namespace,
                graph_top_k=self._graph_top_k,
                graph_max_hops=self._graph_max_hops,
                write_back=self._write_back,
            ),
        )

    def _build_workflow(
        self,
        prompt: str,
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> Workflow:
        """Build the read/reason/write workflow for one resolved run context."""
        del request_id, dependencies
        workflow_steps: list[WorkflowStep] = [
            MemoryReadStep(
                step_id="memory_read",
                query_builder=lambda context: str(context.get("prompt", "")),
                namespace=self._memory_namespace,
                top_k=self._memory_top_k,
                min_score=self._memory_min_score,
            ),
            LogicStep(
                step_id="graph_read",
                handler=lambda context: _build_graph_read_output(
                    graph_memory_store=self._graph_memory_store,
                    query_text=str(context.get("prompt", "")),
                    namespace=self._graph_namespace,
                    top_k=self._graph_top_k,
                    max_hops=self._graph_max_hops,
                    min_score=self._graph_min_score,
                ),
            ),
            DelegateStep(
                step_id="reason",
                dependencies=("memory_read", "graph_read"),
                delegate=self._reasoning_delegate,
                prompt_builder=lambda context: _build_reasoning_prompt(
                    task_prompt=str(context.get("prompt", "")),
                    memory_read_step_output=_extract_dependency_output(
                        context=context,
                        dependency_id="memory_read",
                    ),
                    graph_read_step_output=_extract_dependency_output(
                        context=context,
                        dependency_id="graph_read",
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
        workflow = Workflow(
            tool_runtime=None,
            memory_store=self._memory_store,
            tracer=self._tracer,
            input_schema={"type": "object"},
            base_context={"prompt": prompt},
            steps=workflow_steps,
        )
        self.workflow = workflow
        return workflow

    def _run_rag_pattern(
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
        workflow = self._build_workflow(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        workflow_result = workflow.run(
            input={},
            execution_mode="sequential",
            request_id=f"{request_id}:rag_reasoning",
            dependencies=dependencies,
        )
        return _build_rag_result(
            workflow_result=workflow_result,
            request_id=request_id,
            dependencies=dependencies,
            memory_namespace=self._memory_namespace,
            memory_top_k=self._memory_top_k,
            graph_namespace=self._graph_namespace,
            graph_top_k=self._graph_top_k,
            graph_max_hops=self._graph_max_hops,
            write_back=self._write_back,
        )


def _build_rag_result(
    *,
    workflow_result: ExecutionResult,
    request_id: str,
    dependencies: Mapping[str, object],
    memory_namespace: str,
    memory_top_k: int,
    graph_namespace: str,
    graph_top_k: int,
    graph_max_hops: int,
    write_back: bool,
) -> ExecutionResult:
    """Build final RAG result from one workflow execution."""
    memory_read_result = workflow_result.step_results.get("memory_read")
    graph_read_result = workflow_result.step_results.get("graph_read")
    reason_result = workflow_result.step_results.get("reason")
    memory_write_result = workflow_result.step_results.get("memory_write")

    retrieval_output = (
        dict(memory_read_result.output)
        if memory_read_result is not None
        else {
            "query": {},
            "matches": [],
            "count": 0,
            "namespace": memory_namespace,
        }
    )
    reasoning_output = dict(reason_result.output) if reason_result is not None else {}
    graph_output = (
        dict(graph_read_result.output)
        if graph_read_result is not None
        else {
            "query": {},
            "subgraph": {
                "namespace": graph_namespace,
                "query_text": "",
                "matched_node_ids": [],
                "nodes": [],
                "edges": [],
            },
            "count_nodes": 0,
            "count_edges": 0,
            "namespace": graph_namespace,
        }
    )
    write_back_output = (
        dict(memory_write_result.output)
        if memory_write_result is not None
        else {"written": 0, "namespace": memory_namespace, "ids": []}
    )
    workflow_payload = workflow_result.to_dict()
    workflow_artifacts = workflow_result.output.get("artifacts", [])
    delegate_final_output = reasoning_output.get("output")
    final_output = dict(delegate_final_output) if isinstance(delegate_final_output, Mapping) else dict(reasoning_output)
    retrieval_details = dict(retrieval_output)
    retrieval_details["context"] = _build_retrieval_context(retrieval_output)

    result_success = workflow_result.success
    terminated_reason = "completed" if result_success else "workflow_failure"

    return build_pattern_execution_result(
        success=result_success,
        final_output=final_output,
        terminated_reason=terminated_reason,
        details={
            "retrieval": retrieval_details,
            "graph": graph_output,
            "reasoning": reasoning_output,
            "write_back": write_back_output,
        },
        workflow_payload=workflow_payload,
        artifacts=workflow_artifacts,
        request_id=request_id,
        dependencies=dependencies,
        mode=MODE_RAG,
        metadata={
            "memory_namespace": memory_namespace,
            "memory_top_k": memory_top_k,
            "graph_namespace": graph_namespace,
            "graph_top_k": graph_top_k,
            "graph_max_hops": graph_max_hops,
            "write_back": write_back,
        },
        tool_results=[],
        model_response=None,
        requested_mode=MODE_RAG,
        resolved_mode=MODE_RAG,
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
    graph_read_step_output: Mapping[str, object],
) -> str:
    """Build explicit prompt with retrieved context injection.

    Args:
        task_prompt: Task prompt.
        memory_read_step_output: Output payload from memory read step.
        graph_read_step_output: Output payload from graph read step.

    Returns:
        Prompt string passed to the reasoning delegate.
    """
    context_json_block, context_text = _build_memory_context_blocks(memory_read_step_output)
    graph_json_block, graph_text = _build_graph_context_blocks(graph_read_step_output)

    prompt_lines = [
        f"Task: {task_prompt}",
        "",
        "Retrieved context (JSON):",
        context_json_block,
        "",
        "Retrieved context (text):",
        context_text,
        "",
        "Retrieved graph context (JSON):",
        graph_json_block,
        "",
        "Retrieved graph context (text):",
        graph_text,
        "",
        "Use the retrieved context when relevant, but reason independently when context is sparse.",
    ]
    return "\n".join(prompt_lines)


def _build_memory_context_blocks(memory_read_step_output: Mapping[str, object]) -> tuple[str, str]:
    """Return rendered JSON and text blocks for memory retrieval context."""
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
        content = str(match.get("content", "")).strip()
        if not content:
            continue
        score_text = _format_score(match.get("score"))
        context_text_lines.append(f"- [{item_id}]{score_text} {content}")
    context_text = "\n".join(context_text_lines) if context_text_lines else "(none)"
    return context_json_block, context_text


def _build_graph_context_blocks(graph_read_step_output: Mapping[str, object]) -> tuple[str, str]:
    """Return rendered JSON and text blocks for graph retrieval context."""
    graph_subgraph = graph_read_step_output.get("subgraph")
    normalized_graph_subgraph = dict(graph_subgraph) if isinstance(graph_subgraph, Mapping) else {}
    graph_payload = (
        normalized_graph_subgraph
        if normalized_graph_subgraph
        else {
            "namespace": graph_read_step_output.get("namespace", "default"),
            "matched_node_ids": [],
            "nodes": [],
            "edges": [],
        }
    )
    graph_json_block = json.dumps(
        graph_payload,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    graph_text_lines = _render_graph_text_lines(graph_payload)
    graph_text = "\n".join(graph_text_lines) if graph_text_lines else "(none)"
    return graph_json_block, graph_text


def _render_graph_text_lines(graph_payload: Mapping[str, object]) -> list[str]:
    """Return text lines summarizing graph nodes and edges."""
    graph_text_lines: list[str] = []
    graph_nodes = graph_payload.get("nodes")
    if isinstance(graph_nodes, list):
        for node in graph_nodes:
            if not isinstance(node, Mapping):
                continue
            name = str(node.get("name", node.get("node_id", ""))).strip()
            node_type = str(node.get("node_type", "")).strip()
            if not name:
                continue
            type_suffix = f" ({node_type})" if node_type else ""
            graph_text_lines.append(f"- node: {name}{type_suffix}")
    graph_edges = graph_payload.get("edges")
    if not isinstance(graph_edges, list):
        return graph_text_lines
    for edge in graph_edges:
        if not isinstance(edge, Mapping):
            continue
        source_id = str(edge.get("source_id", "")).strip()
        relationship = str(edge.get("relationship", "")).strip()
        target_id = str(edge.get("target_id", "")).strip()
        if not source_id or not target_id:
            continue
        if relationship:
            graph_text_lines.append(f"- edge: {source_id} -[{relationship}]-> {target_id}")
        else:
            graph_text_lines.append(f"- edge: {source_id} -> {target_id}")
    return graph_text_lines


def _format_score(value: object) -> str:
    """Return one optional score suffix for retrieval text rendering."""
    return f" score={value}" if isinstance(value, (int, float)) else ""


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


def _build_retrieval_context(retrieval_output: Mapping[str, object]) -> dict[str, object]:
    """Return one normalized retrieval-context payload derived from raw matches."""
    matches = retrieval_output.get("matches")
    normalized_matches = (
        [dict(match) for match in matches if isinstance(match, Mapping)] if isinstance(matches, list) else []
    )
    return {
        "namespace": retrieval_output.get("namespace", "default"),
        "count": _safe_int(retrieval_output.get("count")),
        "matches": normalized_matches,
    }


def _build_graph_read_output(
    *,
    graph_memory_store: GraphMemoryStore | None,
    query_text: str,
    namespace: str,
    top_k: int,
    max_hops: int,
    min_score: float | None,
) -> dict[str, object]:
    """Return one normalized graph-retrieval payload for prompt injection."""
    query = GraphSearchQuery(
        text=query_text,
        namespace=namespace,
        top_k=max(1, int(top_k)),
        max_hops=max(0, int(max_hops)),
        min_score=min_score,
    )
    if graph_memory_store is None:
        return {
            "query": query.to_dict(),
            "subgraph": {
                "namespace": namespace,
                "query_text": query_text,
                "matched_node_ids": [],
                "nodes": [],
                "edges": [],
            },
            "count_nodes": 0,
            "count_edges": 0,
            "namespace": namespace,
        }

    try:
        subgraph = graph_memory_store.query_subgraph(query)
    except Exception as exc:
        return {
            "query": query.to_dict(),
            "subgraph": {
                "namespace": namespace,
                "query_text": query_text,
                "matched_node_ids": [],
                "nodes": [],
                "edges": [],
            },
            "count_nodes": 0,
            "count_edges": 0,
            "namespace": namespace,
            "error": str(exc),
        }

    subgraph_payload = subgraph.to_dict()
    return {
        "query": query.to_dict(),
        "subgraph": subgraph_payload,
        "count_nodes": len(subgraph_payload["nodes"]),
        "count_edges": len(subgraph_payload["edges"]),
        "namespace": subgraph.namespace,
    }


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


__all__ = ["RAGPattern"]
