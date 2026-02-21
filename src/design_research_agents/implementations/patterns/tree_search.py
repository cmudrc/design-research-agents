"""Tree-search reasoning pattern with pluggable generator and evaluator delegates."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import TypeGuard, cast

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.workflow import LogicStep, LoopStep, WorkflowDelegate
from design_research_agents.implementations.shared.workflow_internal import (
    execute_pattern_with_trace,
    resolve_pattern_run_context,
)
from design_research_agents.tracing import Tracer
from design_research_agents.workflow import Workflow

from ..shared.workflow_internal.delegate_invocation import invoke_delegate

GeneratorValue = Mapping[str, object] | str | int | float
GeneratorDelegate = Callable[[Mapping[str, object]], Sequence[GeneratorValue]]
EvaluatorDelegate = Callable[[Mapping[str, object]], float | int | Mapping[str, object]]


class TreeSearchPattern(Agent):
    """Beam-style tree search over generated candidate states."""

    def __init__(
        self,
        *,
        generator_delegate: GeneratorDelegate | WorkflowDelegate,
        evaluator_delegate: EvaluatorDelegate | WorkflowDelegate,
        max_depth: int = 3,
        branch_factor: int = 3,
        beam_width: int = 2,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize tree-search reasoning pattern.

        Args:
            generator_delegate: Delegate that expands one candidate into children.
            evaluator_delegate: Delegate that assigns a score to one candidate.
            max_depth: Maximum expansion depth.
            branch_factor: Max children retained per expanded node.
            beam_width: Max frontier width kept after each depth.
            tracer: Optional tracer dependency.

        Raises:
            ValueError: Raised when depth/branch/beam settings are invalid.
        """
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if branch_factor < 1:
            raise ValueError("branch_factor must be >= 1.")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1.")

        self._generator_delegate = _normalize_generator_delegate(generator_delegate)
        self._evaluator_delegate = _normalize_evaluator_delegate(evaluator_delegate)
        self._max_depth = max_depth
        self._branch_factor = branch_factor
        self._beam_width = beam_width
        self._tracer = tracer
        self.workflow: object | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute tree search and return the highest-scoring candidate.

        Args:
            prompt: Task prompt.
            request_id: Optional request identifier.
            dependencies: Optional dependency mapping.

        Returns:
            Tree search result payload.

        Raises:
            Exception: Propagated delegate invocation errors.
        """
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=None,
            default_dependencies={},
            request_id=request_id,
            dependencies=dependencies,
        )
        return execute_pattern_with_trace(
            agent_name="TreeSearchPattern",
            request_id=run_context.request_id,
            input_payload={
                "prompt": prompt,
                "max_depth": self._max_depth,
                "branch_factor": self._branch_factor,
                "beam_width": self._beam_width,
            },
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            runner=lambda: self._run_tree_search(
                prompt=prompt,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def _run_tree_search(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Run beam-style expansion and scoring through workflow loop primitives.

        Args:
            prompt: Task prompt to optimize through tree search.
            request_id: Resolved request id for this run.
            dependencies: Normalized dependency mapping passed to delegates.

        Returns:
            Final tree-search execution result.

        Raises:
            RuntimeError: Raised when the loop step result is missing.
        """
        root_candidate = {"text": prompt, "depth": 0}
        root_node = {
            "node_id": "root",
            "candidate": root_candidate,
            "score": 0.0,
            "depth": 0,
            "parent_id": None,
        }
        initial_state: dict[str, object] = {
            "node_counter": 0,
            "frontier": [root_node],
            "frontier_trace": [],
            "explored_nodes": 0,
            "best_node": root_node,
            "terminated_reason": "max_depth_reached",
            "should_continue": True,
        }

        def _continue_predicate(iteration: int, state: Mapping[str, object]) -> bool:
            """Decide whether another search iteration should run.

            Args:
                iteration: One-based loop iteration index.
                state: Current loop state mapping.

            Returns:
                ``True`` when search should continue.
            """
            del iteration
            return bool(state.get("should_continue", True))

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
            """Execute one expansion/evaluation iteration.

            Args:
                context: Workflow step context for the current iteration.

            Returns:
                Updated loop state for the next iteration.
            """
            loop_meta = context.get("_loop")
            depth = 1
            if isinstance(loop_meta, Mapping):
                depth = max(1, _safe_int(loop_meta.get("iteration", 1)))

            loop_state = context.get("loop_state")
            current_state = dict(loop_state) if isinstance(loop_state, Mapping) else {}
            raw_frontier = current_state.get("frontier")
            frontier = (
                [dict(node) for node in raw_frontier if isinstance(node, Mapping)]
                if isinstance(raw_frontier, list)
                else []
            )
            node_counter = _safe_int(current_state.get("node_counter"))
            frontier_trace = (
                [dict(item) for item in current_state.get("frontier_trace", [])]
                if isinstance(current_state.get("frontier_trace"), list)
                else []
            )
            explored_nodes = _safe_int(current_state.get("explored_nodes"))
            maybe_best_node = current_state.get("best_node")
            best_node = dict(maybe_best_node) if isinstance(maybe_best_node, Mapping) else root_node

            if not frontier:
                return {
                    **current_state,
                    "should_continue": False,
                    "terminated_reason": "no_expansions",
                }

            expanded_nodes: list[dict[str, object]] = []
            for parent_node in frontier:
                children = self._generate_children(
                    prompt=prompt,
                    parent_node=parent_node,
                    depth=depth,
                    request_id=request_id,
                    dependencies=dependencies,
                )
                for child in children[: self._branch_factor]:
                    node_counter += 1
                    child_candidate = _normalize_candidate(child)
                    child_score = self._evaluate_candidate(
                        prompt=prompt,
                        candidate=child_candidate,
                        depth=depth,
                        request_id=request_id,
                        dependencies=dependencies,
                    )
                    explored_nodes += 1
                    expanded_nodes.append(
                        {
                            "node_id": f"node_{node_counter}",
                            "candidate": child_candidate,
                            "score": child_score,
                            "depth": depth,
                            "parent_id": parent_node.get("node_id"),
                        }
                    )

            if not expanded_nodes:
                return {
                    **current_state,
                    "node_counter": node_counter,
                    "explored_nodes": explored_nodes,
                    "should_continue": False,
                    "terminated_reason": "no_expansions",
                }

            expanded_nodes.sort(
                key=lambda node: (
                    _safe_float(node.get("score")),
                    -_safe_int(node.get("depth")),
                    str(node.get("node_id", "")),
                ),
                reverse=True,
            )
            frontier = expanded_nodes[: self._beam_width]
            frontier_trace.append(
                {
                    "depth": depth,
                    "frontier": [
                        {
                            "node_id": node["node_id"],
                            "score": node["score"],
                            "candidate": _json_ready(node["candidate"]),
                            "parent_id": node["parent_id"],
                        }
                        for node in frontier
                    ],
                }
            )
            if frontier and _safe_float(frontier[0].get("score")) >= _safe_float(
                best_node.get("score")
            ):
                best_node = frontier[0]
            should_continue = depth < self._max_depth
            return {
                "node_counter": node_counter,
                "frontier": frontier,
                "frontier_trace": frontier_trace,
                "explored_nodes": explored_nodes,
                "best_node": best_node,
                "should_continue": should_continue,
                "terminated_reason": "looping" if should_continue else "max_depth_reached",
            }

        def _state_reducer(
            state: Mapping[str, object],
            iteration_result: ExecutionResult,
            iteration: int,
        ) -> Mapping[str, object]:
            """Fold one loop iteration output into accumulated search state.

            Args:
                state: Current accumulated loop state.
                iteration_result: Workflow result for one loop iteration.
                iteration: One-based loop iteration index.

            Returns:
                Reduced loop state mapping.
            """
            del iteration
            iteration_step = iteration_result.step_results.get("tree_search_iteration")
            if iteration_step is None or not getattr(iteration_step, "success", False):
                return dict(state)
            output = getattr(iteration_step, "output", {})
            return dict(output) if isinstance(output, Mapping) else dict(state)

        workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
            steps=[
                LoopStep(
                    step_id="tree_search_loop",
                    steps=(
                        LogicStep(
                            step_id="tree_search_iteration",
                            handler=_run_iteration,
                        ),
                    ),
                    max_iterations=self._max_depth,
                    initial_state=initial_state,
                    continue_predicate=_continue_predicate,
                    state_reducer=_state_reducer,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                ),
            ],
        )
        self.workflow = workflow

        workflow_result = workflow.run(
            {},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:tree_search_workflow",
            dependencies=dependencies,
        )
        loop_step_result = workflow_result.step_results.get("tree_search_loop")
        if loop_step_result is None:
            raise RuntimeError("Tree search loop step result is missing.")
        loop_output = loop_step_result.output
        final_state_raw = loop_output.get("final_state")
        final_state = dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}
        best_node_raw = final_state.get("best_node")
        best_node = dict(best_node_raw) if isinstance(best_node_raw, Mapping) else root_node
        best_candidate = _json_ready(best_node.get("candidate", {}))
        best_score = _safe_float(best_node.get("score"))
        explored_nodes = _safe_int(final_state.get("explored_nodes"))
        raw_frontier_trace = final_state.get("frontier_trace")
        frontier_trace = (
            [dict(entry) for entry in raw_frontier_trace if isinstance(entry, Mapping)]
            if isinstance(raw_frontier_trace, list)
            else []
        )

        return ExecutionResult(
            output={
                "best_candidate": best_candidate,
                "best_score": best_score,
                "explored_nodes": explored_nodes,
                "frontier_trace": frontier_trace,
                "final_output": {
                    "best_candidate": best_candidate,
                    "best_score": best_score,
                },
                "workflow": workflow_result.asdict(),
                "artifacts": workflow_result.output.get("artifacts", []),
                "terminated_reason": str(
                    final_state.get(
                        "terminated_reason",
                        loop_output.get("terminated_reason", "completed"),
                    )
                ),
            },
            success=workflow_result.success,
            tool_results=[],
            model_response=None,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "max_depth": self._max_depth,
                "branch_factor": self._branch_factor,
                "beam_width": self._beam_width,
            },
        )

    def _generate_children(
        self,
        *,
        prompt: str,
        parent_node: Mapping[str, object],
        depth: int,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> list[Mapping[str, object] | str | int | float]:
        """Generate child candidates from one parent node.

        Args:
            prompt: Task prompt.
            parent_node: Parent frontier node payload.
            depth: One-based expansion depth.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency mapping.

        Returns:
            Generated child candidates.
        """
        delegate_input = {
            "task": prompt,
            "depth": depth,
            "parent": _json_ready(parent_node.get("candidate", {})),
            "parent_node": _json_ready(parent_node),
            "branch_factor": self._branch_factor,
        }

        delegate_prompt = json.dumps(delegate_input, ensure_ascii=True, sort_keys=True)
        delegate_invocation = invoke_delegate(
            delegate=self._generator_delegate,
            prompt=delegate_prompt,
            step_context=None,
            request_id=f"{request_id}:tree_search:generator:{depth}:{parent_node.get('node_id')}",
            execution_mode="sequential",
            failure_policy="skip_dependents",
            dependencies=dependencies,
        )
        delegate_result = delegate_invocation.result
        if not delegate_result.success:
            return []
        return _extract_candidate_list(delegate_result.output)

    def _evaluate_candidate(
        self,
        *,
        prompt: str,
        candidate: Mapping[str, object],
        depth: int,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> float:
        """Evaluate one candidate and return normalized score.

        Args:
            prompt: Task prompt.
            candidate: Candidate payload.
            depth: One-based candidate depth.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency mapping.

        Returns:
            Candidate score.
        """
        delegate_input = {
            "task": prompt,
            "depth": depth,
            "candidate": _json_ready(candidate),
        }

        delegate_prompt = json.dumps(delegate_input, ensure_ascii=True, sort_keys=True)
        delegate_invocation = invoke_delegate(
            delegate=self._evaluator_delegate,
            prompt=delegate_prompt,
            step_context=None,
            request_id=f"{request_id}:tree_search:evaluator:{depth}",
            execution_mode="sequential",
            failure_policy="skip_dependents",
            dependencies=dependencies,
        )
        delegate_result = delegate_invocation.result
        if not delegate_result.success:
            return 0.0
        return _extract_score(delegate_result.output)


def _normalize_generator_delegate(
    delegate: GeneratorDelegate | WorkflowDelegate,
) -> WorkflowDelegate:
    """Normalize generator delegate into one object-delegate contract.

    Args:
        delegate: Generator callable or workflow-compatible delegate object.

    Returns:
        Workflow-compatible delegate object.
    """
    if _is_workflow_delegate(delegate):
        return delegate
    return _GeneratorCallableDelegateAdapter(cast(GeneratorDelegate, delegate))


def _normalize_evaluator_delegate(
    delegate: EvaluatorDelegate | WorkflowDelegate,
) -> WorkflowDelegate:
    """Normalize evaluator delegate into one object-delegate contract.

    Args:
        delegate: Evaluator callable or workflow-compatible delegate object.

    Returns:
        Workflow-compatible delegate object.
    """
    if _is_workflow_delegate(delegate):
        return delegate
    return _EvaluatorCallableDelegateAdapter(cast(EvaluatorDelegate, delegate))


class _GeneratorCallableDelegateAdapter:
    """Adapter that wraps callable generator delegates as agent-like delegates."""

    def __init__(self, delegate: GeneratorDelegate) -> None:
        """Store callable generator delegate.

        Args:
            delegate: Callable generator delegate.
        """
        self._delegate = delegate

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute callable generator delegate and normalize output.

        Args:
            prompt: Serialized delegate context payload.
            request_id: Optional request identifier.
            dependencies: Optional dependency mapping.

        Returns:
            Normalized execution result containing ``candidates`` output.
        """
        del request_id, dependencies
        context = _parse_json_context(prompt)
        try:
            raw_children = self._delegate(context)
        except Exception as exc:
            return ExecutionResult(
                output={"error": str(exc)},
                success=False,
                tool_results=[],
                model_response=None,
                metadata={"delegate_type": "callable_generator"},
            )
        return ExecutionResult(
            output={"candidates": list(raw_children)},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate_type": "callable_generator"},
        )


class _EvaluatorCallableDelegateAdapter:
    """Adapter that wraps callable evaluator delegates as agent-like delegates."""

    def __init__(self, delegate: EvaluatorDelegate) -> None:
        """Store callable evaluator delegate.

        Args:
            delegate: Callable evaluator delegate.
        """
        self._delegate = delegate

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute callable evaluator delegate and normalize score output.

        Args:
            prompt: Serialized delegate context payload.
            request_id: Optional request identifier.
            dependencies: Optional dependency mapping.

        Returns:
            Normalized execution result containing a numeric score payload.
        """
        del request_id, dependencies
        context = _parse_json_context(prompt)
        try:
            raw_score = self._delegate(context)
        except Exception as exc:
            return ExecutionResult(
                output={"error": str(exc)},
                success=False,
                tool_results=[],
                model_response=None,
                metadata={"delegate_type": "callable_evaluator"},
            )
        if isinstance(raw_score, (int, float)):
            output: dict[str, object] = {"score": float(raw_score)}
        elif isinstance(raw_score, Mapping):
            output = dict(raw_score)
        else:
            output = {"score": 0.0}
        return ExecutionResult(
            output=output,
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate_type": "callable_evaluator"},
        )


def _parse_json_context(prompt: str) -> dict[str, object]:
    """Parse delegate prompt payload into mapping context.

    Args:
        prompt: Serialized JSON payload or raw prompt text.

    Returns:
        Mapping context for callable delegate invocation.
    """
    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        return {"task": prompt}
    if isinstance(parsed, Mapping):
        return {str(key): _json_ready(value) for key, value in parsed.items()}
    return {"task": prompt}


def _is_workflow_delegate(delegate: object) -> TypeGuard[WorkflowDelegate]:
    """Return whether delegate appears to implement workflow-delegate contract.

    Args:
        delegate: Delegate candidate object.

    Returns:
        ``True`` when ``delegate`` exposes a callable ``run`` method.
    """
    run_callable = getattr(delegate, "run", None)
    return callable(run_callable)


def _is_agent_like(delegate: object) -> bool:
    """Return whether delegate appears to implement an agent-like ``run`` method.

    Args:
        delegate: Delegate candidate object.

    Returns:
        ``True`` when the delegate has a callable ``run`` attribute.
    """
    return _is_workflow_delegate(delegate)


def _extract_candidate_list(output: Mapping[str, object]) -> list[GeneratorValue]:
    """Extract candidate list from delegate output payload.

    Args:
        output: Delegate output mapping.

    Returns:
        Normalized candidate list.
    """
    candidates = output.get("candidates")
    if isinstance(candidates, list):
        return _coerce_generator_values(candidates)

    candidate = output.get("candidate")
    normalized_candidate = _normalize_generator_value(candidate)
    if normalized_candidate is not None:
        return [normalized_candidate]

    model_text = output.get("model_text")
    if isinstance(model_text, str):
        try:
            parsed = json.loads(model_text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return _coerce_generator_values(parsed)
        if isinstance(parsed, Mapping):
            parsed_candidates = parsed.get("candidates")
            if isinstance(parsed_candidates, list):
                return _coerce_generator_values(parsed_candidates)
            parsed_candidate = parsed.get("candidate")
            normalized_parsed_candidate = _normalize_generator_value(parsed_candidate)
            if normalized_parsed_candidate is not None:
                return [normalized_parsed_candidate]
    return []


def _extract_score(output: Mapping[str, object]) -> float:
    """Extract numeric score from delegate output payload.

    Args:
        output: Delegate output mapping.

    Returns:
        Extracted numeric score.
    """
    score = output.get("score")
    if isinstance(score, (int, float)):
        return float(score)

    model_text = output.get("model_text")
    if isinstance(model_text, str):
        try:
            parsed = json.loads(model_text)
        except json.JSONDecodeError:
            return 0.0
        if isinstance(parsed, Mapping):
            parsed_score = parsed.get("score")
            if isinstance(parsed_score, (int, float)):
                return float(parsed_score)
    return 0.0


def _normalize_candidate(
    candidate: Mapping[str, object] | str | int | float,
) -> dict[str, object]:
    """Normalize one candidate payload to a mapping.

    Args:
        candidate: Candidate payload.

    Returns:
        Mapping-form candidate payload.
    """
    if isinstance(candidate, Mapping):
        return {str(key): _json_ready(value) for key, value in candidate.items()}
    return {"value": _json_ready(candidate)}


def _normalize_generator_value(value: object) -> GeneratorValue | None:
    """Normalize raw candidate value to supported generator union.

    Args:
        value: Raw candidate value.

    Returns:
        Normalized value when supported, otherwise ``None``.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (str, int, float)):
        return value
    return None


def _coerce_generator_values(values: Sequence[object]) -> list[GeneratorValue]:
    """Coerce heterogeneous values to supported generator union values.

    Args:
        values: Raw candidate values.

    Returns:
        Supported candidate values only.
    """
    normalized: list[GeneratorValue] = []
    for value in values:
        normalized_value = _normalize_generator_value(value)
        if normalized_value is not None:
            normalized.append(normalized_value)
    return normalized


def _safe_float(value: object) -> float:
    """Convert values to float with deterministic fallback to zero.

    Args:
        value: Raw input value.

    Returns:
        Float representation or ``0.0`` fallback.
    """
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


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


def _json_ready(value: object) -> object:
    """Recursively convert values into JSON-safe shapes.

    Args:
        value: Raw input value.

    Returns:
        JSON-safe representation.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["TreeSearchPattern"]
