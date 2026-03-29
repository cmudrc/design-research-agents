"""Tree-search reasoning pattern with pluggable generator and evaluator delegates."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._workflow import DelegateTarget, LogicStep, LoopStep
from design_research_agents._implementations._patterns._tree_search_delegate_adapters import (
    EvaluatorCallableDelegateAdapter,
    GeneratorCallableDelegateAdapter,
    is_workflow_delegate,
)
from design_research_agents._runtime._common._delegate_invocation import invoke_delegate
from design_research_agents._runtime._patterns import (
    MODE_TREE_SEARCH,
    LoopCallbacks,
    build_compiled_pattern_execution,
    build_loop_callbacks,
    build_pattern_execution_result,
    resolve_pattern_run_context,
    wrap_iteration_handler,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution, Workflow

GeneratorValue = Mapping[str, object] | str | int | float
GeneratorDelegate = Callable[[Mapping[str, object]], Sequence[GeneratorValue]]
EvaluatorDelegate = Callable[[Mapping[str, object]], float | int | Mapping[str, object]]

SearchStrategy = Literal["beam", "mcts"]


class TreeSearchPattern(Delegate):
    """Tree-search pattern with beam and MCTS strategies."""

    def __init__(
        self,
        *,
        generator_delegate: GeneratorDelegate | DelegateTarget,
        evaluator_delegate: EvaluatorDelegate | DelegateTarget,
        max_depth: int = 3,
        branch_factor: int = 3,
        beam_width: int = 2,
        search_strategy: SearchStrategy = "beam",
        mcts_exploration_weight: float = 1.4,
        simulation_budget: int | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize tree-search reasoning pattern."""
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if branch_factor < 1:
            raise ValueError("branch_factor must be >= 1.")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1.")
        if search_strategy not in {"beam", "mcts"}:
            raise ValueError("search_strategy must be one of: beam, mcts.")
        if mcts_exploration_weight <= 0:
            raise ValueError("mcts_exploration_weight must be > 0.")
        if simulation_budget is not None and simulation_budget < 1:
            raise ValueError("simulation_budget must be >= 1 when provided.")

        self._generator_delegate = _normalize_generator_delegate(generator_delegate)
        self._evaluator_delegate = _normalize_evaluator_delegate(evaluator_delegate)
        self._max_depth = max_depth
        self._branch_factor = branch_factor
        self._beam_width = beam_width
        self._search_strategy = search_strategy
        self._mcts_exploration_weight = mcts_exploration_weight
        self._simulation_budget = simulation_budget
        self._tracer = tracer
        self.workflow: object | None = None
        self._tree_search_runtime: dict[str, object] | None = None

    def run(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute tree search and return the highest-scoring candidate."""
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
        """Compile one tree-search workflow."""
        run_context = resolve_pattern_run_context(
            prompt=prompt,
            default_request_id_prefix=None,
            default_dependencies={},
            request_id=request_id,
            dependencies=dependencies,
        )
        resolved_simulation_budget = self._resolve_simulation_budget()
        input_payload = {
            **run_context.normalized_input,
            "mode": MODE_TREE_SEARCH,
            "search_strategy": self._search_strategy,
            "max_depth": self._max_depth,
            "branch_factor": self._branch_factor,
            "beam_width": self._beam_width,
            "mcts_exploration_weight": self._mcts_exploration_weight,
            "simulation_budget": resolved_simulation_budget,
        }
        workflow = self._build_workflow(
            run_context.prompt,
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
        )
        runtime = self._tree_search_runtime or {}
        maybe_root_node = runtime.get("root_node")
        root_node = (
            dict(maybe_root_node)
            if isinstance(maybe_root_node, Mapping)
            else {
                "node_id": "root",
                "candidate": {"text": run_context.prompt, "depth": 0},
                "score": 0.0,
                "depth": 0,
                "parent_id": None,
            }
        )
        return build_compiled_pattern_execution(
            workflow=workflow,
            pattern_name="TreeSearchPattern",
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            input_payload=input_payload,
            workflow_request_id=f"{run_context.request_id}:tree_search_workflow",
            finalize=lambda workflow_result: _build_tree_search_result(
                workflow_result=workflow_result,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
                max_depth=self._max_depth,
                branch_factor=self._branch_factor,
                beam_width=self._beam_width,
                search_strategy=self._search_strategy,
                mcts_exploration_weight=self._mcts_exploration_weight,
                simulation_budget=resolved_simulation_budget,
                root_node=root_node,
            ),
        )

    def _resolve_simulation_budget(self) -> int:
        if self._simulation_budget is not None:
            return self._simulation_budget
        return max(1, self._max_depth * max(self._branch_factor, self._beam_width))

    def _build_workflow(
        self,
        prompt: str,
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> Workflow:
        """Build the tree-search workflow for one resolved run context."""
        root_candidate = {"text": prompt, "depth": 0}
        root_node = {
            "node_id": "root",
            "candidate": root_candidate,
            "score": 0.0,
            "depth": 0,
            "parent_id": None,
            "visits": 0,
            "total_score": 0.0,
            "children": [],
            "expanded": False,
            "terminal": False,
        }

        if self._search_strategy == "beam":
            loop_callbacks, initial_state, max_iterations = self._build_beam_loop(
                prompt=prompt,
                request_id=request_id,
                dependencies=dependencies,
                root_node=root_node,
            )
        else:
            loop_callbacks, initial_state, max_iterations = self._build_mcts_loop(
                prompt=prompt,
                request_id=request_id,
                dependencies=dependencies,
                root_node=root_node,
            )

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
                            handler=loop_callbacks.iteration_handler,
                        ),
                    ),
                    max_iterations=max_iterations,
                    initial_state=initial_state,
                    continue_predicate=loop_callbacks.continue_predicate,
                    state_reducer=loop_callbacks.state_reducer,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                ),
            ],
        )
        self.workflow = workflow
        self._tree_search_runtime = {"root_node": dict(root_node)}
        return workflow

    def _build_beam_loop(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        root_node: Mapping[str, object],
    ) -> tuple[LoopCallbacks, dict[str, object], int]:
        """Build beam-search loop callbacks and state."""
        initial_state: dict[str, object] = {
            "node_counter": 0,
            "frontier": [dict(root_node)],
            "frontier_trace": [],
            "explored_nodes": 0,
            "best_node": dict(root_node),
            "terminated_reason": "max_depth_reached",
            "should_continue": True,
        }

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
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
            best_node = dict(maybe_best_node) if isinstance(maybe_best_node, Mapping) else dict(root_node)

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
            if frontier and _safe_float(frontier[0].get("score")) >= _safe_float(best_node.get("score")):
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

        wrapped_handler = wrap_iteration_handler(
            _run_iteration,
            error_prefix="TreeSearchPattern beam iteration",
        )
        loop_callbacks = build_loop_callbacks(
            iteration_step_id="tree_search_iteration",
            iteration_handler=wrapped_handler,
        )
        return loop_callbacks, initial_state, self._max_depth

    def _build_mcts_loop(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        root_node: Mapping[str, object],
    ) -> tuple[LoopCallbacks, dict[str, object], int]:
        """Build MCTS loop callbacks and state."""
        simulation_budget = self._resolve_simulation_budget()
        initial_state: dict[str, object] = {
            "node_counter": 0,
            "nodes": {"root": dict(root_node)},
            "frontier_trace": [],
            "explored_nodes": 0,
            "best_node": dict(root_node),
            "simulations_run": 0,
            "terminated_reason": "simulation_budget_reached",
            "should_continue": True,
        }

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
            loop_state = context.get("loop_state")
            current_state = dict(loop_state) if isinstance(loop_state, Mapping) else {}
            nodes = _normalize_nodes(current_state.get("nodes"))
            if "root" not in nodes:
                nodes["root"] = dict(root_node)

            node_counter = _safe_int(current_state.get("node_counter"))
            explored_nodes = _safe_int(current_state.get("explored_nodes"))
            simulations_run = _safe_int(current_state.get("simulations_run"))
            frontier_trace = (
                [dict(item) for item in current_state.get("frontier_trace", [])]
                if isinstance(current_state.get("frontier_trace"), list)
                else []
            )
            maybe_best_node = current_state.get("best_node")
            best_node = dict(maybe_best_node) if isinstance(maybe_best_node, Mapping) else dict(root_node)

            selected_path = _select_mcts_path(
                nodes=nodes,
                max_depth=self._max_depth,
                exploration_weight=self._mcts_exploration_weight,
            )
            if not selected_path:
                return {
                    **current_state,
                    "should_continue": False,
                    "terminated_reason": "no_expansions",
                    "nodes": _json_ready(nodes),
                }

            selected_node_id = selected_path[-1]
            selected_node = nodes[selected_node_id]
            selected_depth = _safe_int(selected_node.get("depth"))
            rollout_score = _safe_float(selected_node.get("score"))
            backprop_path = list(selected_path)

            if selected_depth >= self._max_depth:
                selected_node["terminal"] = True
            else:
                children = self._generate_children(
                    prompt=prompt,
                    parent_node=selected_node,
                    depth=(selected_depth + 1),
                    request_id=request_id,
                    dependencies=dependencies,
                )
                expanded_child_ids: list[str] = []
                for child in children[: self._branch_factor]:
                    node_counter += 1
                    child_id = f"node_{node_counter}"
                    child_candidate = _normalize_candidate(child)
                    child_score = self._evaluate_candidate(
                        prompt=prompt,
                        candidate=child_candidate,
                        depth=(selected_depth + 1),
                        request_id=request_id,
                        dependencies=dependencies,
                    )
                    explored_nodes += 1
                    child_node = {
                        "node_id": child_id,
                        "candidate": child_candidate,
                        "score": child_score,
                        "depth": selected_depth + 1,
                        "parent_id": selected_node_id,
                        "visits": 0,
                        "total_score": 0.0,
                        "children": [],
                        "expanded": False,
                        "terminal": selected_depth + 1 >= self._max_depth,
                    }
                    nodes[child_id] = child_node
                    expanded_child_ids.append(child_id)

                if expanded_child_ids:
                    selected_node["children"] = expanded_child_ids
                    selected_node["expanded"] = True
                    chosen_child_id = max(
                        expanded_child_ids,
                        key=lambda node_id: (
                            _safe_float(nodes[node_id].get("score")),
                            -_safe_int(nodes[node_id].get("depth")),
                            str(node_id),
                        ),
                    )
                    rollout_score = _safe_float(nodes[chosen_child_id].get("score"))
                    backprop_path.append(chosen_child_id)
                else:
                    selected_node["expanded"] = True
                    selected_node["terminal"] = True

            for node_id in backprop_path:
                node = nodes.get(node_id)
                if not isinstance(node, Mapping):
                    continue
                visits = _safe_int(node.get("visits")) + 1
                total_score = _safe_float(node.get("total_score")) + rollout_score
                mutable_node = dict(node)
                mutable_node["visits"] = visits
                mutable_node["total_score"] = total_score
                nodes[node_id] = mutable_node

            best_node = _select_best_node(nodes=nodes, fallback=best_node)
            simulations_run += 1
            should_continue = simulations_run < simulation_budget and _has_expandable_nodes(
                nodes=nodes,
                max_depth=self._max_depth,
            )
            terminated_reason = "looping"
            if not should_continue:
                terminated_reason = (
                    "simulation_budget_reached" if simulations_run >= simulation_budget else "no_expansions"
                )

            frontier_trace.append(
                {
                    "simulation": simulations_run,
                    "selected_path": list(backprop_path),
                    "top_nodes": _build_mcts_top_nodes(nodes=nodes, top_k=self._beam_width),
                }
            )

            return {
                "node_counter": node_counter,
                "nodes": _json_ready(nodes),
                "frontier_trace": frontier_trace,
                "explored_nodes": explored_nodes,
                "best_node": _json_ready(best_node),
                "simulations_run": simulations_run,
                "should_continue": should_continue,
                "terminated_reason": terminated_reason,
            }

        wrapped_handler = wrap_iteration_handler(
            _run_iteration,
            error_prefix="TreeSearchPattern MCTS iteration",
        )
        loop_callbacks = build_loop_callbacks(
            iteration_step_id="tree_search_iteration",
            iteration_handler=wrapped_handler,
        )
        return loop_callbacks, initial_state, simulation_budget

    def _generate_children(
        self,
        *,
        prompt: str,
        parent_node: Mapping[str, object],
        depth: int,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> list[Mapping[str, object] | str | int | float]:
        """Generate child candidates from one parent node."""
        delegate_input = {
            "task": prompt,
            "depth": depth,
            "parent": _json_ready(parent_node.get("candidate", {})),
            "parent_node": _json_ready(parent_node),
            "branch_factor": self._branch_factor,
            "search_strategy": self._search_strategy,
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
        """Evaluate one candidate and return normalized score."""
        delegate_input = {
            "task": prompt,
            "depth": depth,
            "candidate": _json_ready(candidate),
            "search_strategy": self._search_strategy,
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


def _build_tree_search_result(
    *,
    workflow_result: ExecutionResult,
    request_id: str,
    dependencies: Mapping[str, object],
    max_depth: int,
    branch_factor: int,
    beam_width: int,
    search_strategy: SearchStrategy,
    mcts_exploration_weight: float,
    simulation_budget: int,
    root_node: Mapping[str, object],
) -> ExecutionResult:
    """Build the final tree-search result from one workflow execution."""
    loop_step_result = workflow_result.step_results.get("tree_search_loop")
    if loop_step_result is None:
        raise RuntimeError("Tree search loop step result is missing.")

    loop_output = loop_step_result.output
    final_state_raw = loop_output.get("final_state")
    final_state = dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}

    best_node_raw = final_state.get("best_node")
    best_node = dict(best_node_raw) if isinstance(best_node_raw, Mapping) else dict(root_node)
    best_candidate_raw = best_node.get("candidate")
    best_candidate = (
        dict(best_candidate_raw)
        if isinstance(best_candidate_raw, Mapping)
        else {"value": _json_ready(best_candidate_raw)}
    )
    best_score = _safe_float(best_node.get("score"))

    raw_frontier_trace = final_state.get("frontier_trace")
    frontier_trace = (
        [dict(item) for item in raw_frontier_trace if isinstance(item, Mapping)]
        if isinstance(raw_frontier_trace, list)
        else []
    )
    explored_nodes = _safe_int(final_state.get("explored_nodes"))
    simulations_run = _safe_int(final_state.get("simulations_run"))
    terminated_reason = str(
        final_state.get(
            "terminated_reason",
            loop_output.get("terminated_reason", "max_depth_reached"),
        )
    )
    success = bool(loop_output.get("success", loop_step_result.success))

    details: dict[str, object] = {
        "search_strategy": search_strategy,
        "best_node": _json_ready(best_node),
        "explored_nodes": explored_nodes,
        "frontier_trace": [_json_ready(item) for item in frontier_trace],
    }
    if search_strategy == "mcts":
        details["simulations_run"] = simulations_run
        details["simulation_budget"] = simulation_budget

    return build_pattern_execution_result(
        success=success,
        final_output={
            "best_candidate": _json_ready(best_candidate),
            "best_score": best_score,
        },
        terminated_reason=terminated_reason,
        details=details,
        workflow_payload=workflow_result.to_dict(),
        artifacts=workflow_result.output.get("artifacts", []),
        request_id=request_id,
        dependencies=dependencies,
        mode=MODE_TREE_SEARCH,
        metadata={
            "max_depth": max_depth,
            "branch_factor": branch_factor,
            "beam_width": beam_width,
            "search_strategy": search_strategy,
            "mcts_exploration_weight": mcts_exploration_weight,
            "simulation_budget": simulation_budget,
        },
        error=loop_step_result.error,
    )


def _normalize_nodes(raw_nodes: object) -> dict[str, dict[str, object]]:
    """Normalize stored node mapping payload."""
    if not isinstance(raw_nodes, Mapping):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for node_id, raw_node in raw_nodes.items():
        if not isinstance(node_id, str) or not isinstance(raw_node, Mapping):
            continue
        normalized[node_id] = {str(key): _json_ready(value) for key, value in raw_node.items()}
    return normalized


def _select_best_node(
    *,
    nodes: Mapping[str, Mapping[str, object]],
    fallback: Mapping[str, object],
) -> dict[str, object]:
    """Select best node by score using deterministic ties."""
    if not nodes:
        return dict(fallback)
    best_node_id = max(
        nodes,
        key=lambda node_id: (
            _safe_float(nodes[node_id].get("score")),
            -_safe_int(nodes[node_id].get("depth")),
            str(node_id),
        ),
    )
    selected = nodes.get(best_node_id)
    if isinstance(selected, Mapping):
        return dict(selected)
    return dict(fallback)


def _has_expandable_nodes(*, nodes: Mapping[str, Mapping[str, object]], max_depth: int) -> bool:
    """Return whether any node can still be expanded."""
    for node in nodes.values():
        depth = _safe_int(node.get("depth"))
        if depth >= max_depth:
            continue
        if bool(node.get("terminal", False)):
            continue
        if not bool(node.get("expanded", False)):
            return True
        child_ids = node.get("children")
        if isinstance(child_ids, list):
            for child_id in child_ids:
                child_node = nodes.get(str(child_id))
                if isinstance(child_node, Mapping) and not bool(child_node.get("terminal", False)):
                    return True
    return False


def _select_mcts_path(
    *,
    nodes: Mapping[str, Mapping[str, object]],
    max_depth: int,
    exploration_weight: float,
) -> list[str]:
    """Select one MCTS traversal path with deterministic tie breaks."""
    if "root" not in nodes:
        return []

    path = ["root"]
    current_id = "root"
    visited: set[str] = {"root"}

    while True:
        current_node = nodes.get(current_id)
        if not isinstance(current_node, Mapping):
            return []
        current_depth = _safe_int(current_node.get("depth"))
        if current_depth >= max_depth:
            return path

        expanded = bool(current_node.get("expanded", False))
        raw_children = current_node.get("children")
        child_ids = [str(child_id) for child_id in raw_children] if isinstance(raw_children, list) else []
        child_ids = [child_id for child_id in child_ids if child_id in nodes]
        if not expanded or not child_ids:
            return path

        parent_visits = max(1, _safe_int(current_node.get("visits")))
        next_id = max(
            child_ids,
            key=lambda child_id: (
                _mcts_ucb(
                    node=nodes[child_id],
                    parent_visits=parent_visits,
                    exploration_weight=exploration_weight,
                ),
                -_safe_int(nodes[child_id].get("depth")),
                str(child_id),
            ),
        )
        if next_id in visited:
            return path
        path.append(next_id)
        visited.add(next_id)
        current_id = next_id


def _mcts_ucb(*, node: Mapping[str, object], parent_visits: int, exploration_weight: float) -> float:
    """Return deterministic UCB score for one node."""
    visits = _safe_int(node.get("visits"))
    if visits <= 0:
        return float("inf")
    avg_value = _safe_float(node.get("total_score")) / visits
    exploration_term = exploration_weight * math.sqrt(math.log(parent_visits + 1) / visits)
    return avg_value + exploration_term


def _build_mcts_top_nodes(*, nodes: Mapping[str, Mapping[str, object]], top_k: int) -> list[dict[str, object]]:
    """Return top-ranked nodes for trace snapshots."""
    ranked = sorted(
        (dict(node) for node in nodes.values() if isinstance(node, Mapping) and str(node.get("node_id", ""))),
        key=lambda node: (
            _safe_float(node.get("score")),
            _safe_int(node.get("visits")),
            -_safe_int(node.get("depth")),
            str(node.get("node_id", "")),
        ),
        reverse=True,
    )
    selected = ranked[: max(1, top_k)]
    return [
        {
            "node_id": str(node.get("node_id", "")),
            "score": _safe_float(node.get("score")),
            "visits": _safe_int(node.get("visits")),
            "depth": _safe_int(node.get("depth")),
        }
        for node in selected
    ]


def _normalize_generator_delegate(
    delegate: GeneratorDelegate | DelegateTarget,
) -> DelegateTarget:
    """Normalize generator delegate into one object-delegate contract."""
    if is_workflow_delegate(delegate):
        return delegate
    return GeneratorCallableDelegateAdapter(cast(GeneratorDelegate, delegate))


def _normalize_evaluator_delegate(
    delegate: EvaluatorDelegate | DelegateTarget,
) -> DelegateTarget:
    """Normalize evaluator delegate into one object-delegate contract."""
    if is_workflow_delegate(delegate):
        return delegate
    return EvaluatorCallableDelegateAdapter(cast(EvaluatorDelegate, delegate))


def _is_agent_like(delegate: object) -> bool:
    """Return whether delegate appears to implement an agent-like ``run`` method."""
    return is_workflow_delegate(delegate)


def _extract_candidate_list(output: Mapping[str, object]) -> list[GeneratorValue]:
    """Extract candidate list from delegate output payload."""
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
    """Extract numeric score from delegate output payload."""
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
    """Normalize one candidate payload to a mapping."""
    if isinstance(candidate, Mapping):
        return {str(key): _json_ready(value) for key, value in candidate.items()}
    return {"value": _json_ready(candidate)}


def _normalize_generator_value(value: object) -> GeneratorValue | None:
    """Normalize raw candidate value to supported generator union."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (str, int, float)):
        return value
    return None


def _coerce_generator_values(values: Sequence[object]) -> list[GeneratorValue]:
    """Coerce heterogeneous values to supported generator union values."""
    normalized: list[GeneratorValue] = []
    for value in values:
        normalized_value = _normalize_generator_value(value)
        if normalized_value is not None:
            normalized.append(normalized_value)
    return normalized


def _safe_float(value: object) -> float:
    """Convert values to float with deterministic fallback to zero."""
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
    """Convert values to int with deterministic fallback to zero."""
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
    """Recursively convert values into JSON-safe shapes."""
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
