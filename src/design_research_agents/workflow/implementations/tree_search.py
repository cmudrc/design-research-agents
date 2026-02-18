"""Tree-search reasoning pattern with pluggable generator and evaluator delegates."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    resolve_request_id,
)
from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.tracing import Tracer, finish_trace_run, start_trace_run

GeneratorValue = Mapping[str, object] | str | int | float
GeneratorDelegate = Callable[[Mapping[str, object]], Sequence[GeneratorValue]]
EvaluatorDelegate = Callable[[Mapping[str, object]], float | int | Mapping[str, object]]


class TreeSearchPattern(Agent):
    """Beam-style tree search over generated candidate states."""

    def __init__(
        self,
        *,
        generator_delegate: GeneratorDelegate | Agent,
        evaluator_delegate: EvaluatorDelegate | Agent,
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

        self._generator_delegate = generator_delegate
        self._evaluator_delegate = evaluator_delegate
        self._max_depth = max_depth
        self._branch_factor = branch_factor
        self._beam_width = beam_width
        self._tracer = tracer

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
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        trace_scope = start_trace_run(
            agent_name="TreeSearchPattern",
            request_id=resolved_request_id,
            input_payload={
                "prompt": prompt,
                "max_depth": self._max_depth,
                "branch_factor": self._branch_factor,
                "beam_width": self._beam_width,
            },
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )

        try:
            result = self._run_tree_search(
                prompt=prompt,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
            )
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        finish_trace_run(trace_scope, result=result)
        return result

    def _run_tree_search(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Run beam-style expansion and scoring.

        Args:
            prompt: Task prompt.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency mapping.

        Returns:
            Aggregated tree search result.
        """
        node_counter = 0
        root_candidate = {"text": prompt, "depth": 0}
        frontier: list[dict[str, object]] = [
            {
                "node_id": "root",
                "candidate": root_candidate,
                "score": 0.0,
                "depth": 0,
                "parent_id": None,
            }
        ]
        frontier_trace: list[dict[str, object]] = []
        explored_nodes = 0
        best_node = frontier[0]

        for depth in range(1, self._max_depth + 1):
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
                break

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

        output = {
            "best_candidate": _json_ready(best_node.get("candidate", {})),
            "best_score": _safe_float(best_node.get("score")),
            "explored_nodes": explored_nodes,
            "frontier_trace": frontier_trace,
        }
        return ExecutionResult(
            output=output,
            success=True,
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

        if _is_agent_like(self._generator_delegate):
            delegate_agent = cast(Agent, self._generator_delegate)
            delegate_prompt = json.dumps(delegate_input, ensure_ascii=True, sort_keys=True)
            delegate_result = delegate_agent.run(
                delegate_prompt,
                request_id=f"{request_id}:tree_search:generator:{depth}:{parent_node.get('node_id')}",
                dependencies=dependencies,
            )
            if not delegate_result.success:
                return []
            return _extract_candidate_list(delegate_result.output)

        generator_delegate = cast(GeneratorDelegate, self._generator_delegate)
        raw_children = generator_delegate(delegate_input)
        return list(raw_children)

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

        if _is_agent_like(self._evaluator_delegate):
            delegate_agent = cast(Agent, self._evaluator_delegate)
            delegate_prompt = json.dumps(delegate_input, ensure_ascii=True, sort_keys=True)
            delegate_result = delegate_agent.run(
                delegate_prompt,
                request_id=f"{request_id}:tree_search:evaluator:{depth}",
                dependencies=dependencies,
            )
            if not delegate_result.success:
                return 0.0
            return _extract_score(delegate_result.output)

        evaluator_delegate = cast(EvaluatorDelegate, self._evaluator_delegate)
        raw_score = evaluator_delegate(delegate_input)
        if isinstance(raw_score, (int, float)):
            return float(raw_score)
        if isinstance(raw_score, Mapping):
            return _extract_score(raw_score)
        return 0.0


def _is_agent_like(delegate: object) -> bool:
    """Return whether delegate appears to implement the ``Agent`` contract.

    Args:
        delegate: Delegate candidate object.

    Returns:
        ``True`` when ``delegate`` appears agent-like.
    """
    run_callable = getattr(delegate, "run", None)
    return callable(run_callable)


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
