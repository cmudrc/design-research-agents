"""Networked and blackboard workflow patterns without central orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.workflow import LogicStep, LoopStep, WorkflowDelegate
from design_research_agents.implementations.shared.agent_internal.run_options import (
    normalize_dependencies,
    resolve_request_id,
)
from design_research_agents.tracing import Tracer, finish_trace_run, start_trace_run
from design_research_agents.workflow import Workflow

from ..shared.workflow_internal.delegate_invocation import invoke_delegate


class NetworkedPattern(Agent):
    """Round-based peer coordination pattern with deterministic peer ordering."""

    def __init__(
        self,
        *,
        peers: Mapping[str, WorkflowDelegate],
        max_rounds: int = 4,
        initial_state: Mapping[str, object] | None = None,
        peer_prompt_builder: (Callable[[str, Mapping[str, object], str, int], str] | None) = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize peer-only networked orchestration.

        Args:
            peers: Mapping of peer ids to delegate objects.
            max_rounds: Maximum number of coordination rounds.
            initial_state: Optional initial shared state payload.
            peer_prompt_builder: Optional prompt builder per peer and round.
            tracer: Optional tracer dependency.

        Raises:
            ValueError: Raised when peers is empty or max_rounds is invalid.
        """
        normalized_peers = {
            peer_id.strip(): peer
            for peer_id, peer in peers.items()
            if isinstance(peer_id, str) and peer_id.strip()
        }
        if not normalized_peers:
            raise ValueError("peers must include at least one non-empty peer id.")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1.")

        self._peers = normalized_peers
        self._max_rounds = max_rounds
        self._peer_prompt_builder = peer_prompt_builder or _default_peer_prompt_builder
        self._tracer = tracer
        self._initial_state = dict(initial_state or {})
        self.workflow: object | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute peer-only networked coordination rounds.

        Args:
            prompt: Task prompt shared across peers.
            request_id: Optional request id.
            dependencies: Optional dependency mapping.

        Returns:
            Final pattern result with shared state and round summaries.

        Raises:
            Exception: Propagated peer or reducer failures.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        trace_scope = start_trace_run(
            agent_name=self.__class__.__name__,
            request_id=resolved_request_id,
            input_payload={"prompt": prompt, "max_rounds": self._max_rounds},
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )

        try:
            result = self._run_network(
                prompt=prompt,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
            )
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        finish_trace_run(trace_scope, result=result)
        return result

    def _run_network(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Execute the core peer coordination loop via workflow primitives.

        Args:
            prompt: Task prompt shared across peers.
            request_id: Resolved request id for this run.
            dependencies: Normalized dependency mapping passed to peers.

        Returns:
            Final execution result produced by the network workflow.
        """
        peer_ids = sorted(self._peers)
        initial_state: dict[str, object] = {
            "blackboard": self._initial_blackboard(prompt),
            "round_summaries": [],
            "terminated_reason": "max_rounds_reached",
            "should_continue": True,
            "success": True,
            "failed_peer": None,
            "peer_output": {},
            "tool_results": [],
            "last_model_response": None,
        }

        def _continue_predicate(iteration: int, state: Mapping[str, object]) -> bool:
            """Decide whether the loop should continue.

            Args:
                iteration: One-based loop iteration index.
                state: Current loop state mapping.

            Returns:
                ``True`` when another network round should execute.
            """
            del iteration
            return bool(state.get("should_continue", True))

        def _run_round(context: Mapping[str, object]) -> Mapping[str, object]:
            """Execute one full network round across all peers.

            Args:
                context: Workflow step context for this iteration.

            Returns:
                Updated loop state for the next iteration.
            """
            loop_meta = context.get("_loop")
            round_number = 1
            if isinstance(loop_meta, Mapping):
                round_number = max(1, _safe_int(loop_meta.get("iteration", 1)))

            loop_state = context.get("loop_state")
            current_state = dict(loop_state) if isinstance(loop_state, Mapping) else {}
            raw_blackboard = current_state.get("blackboard")
            blackboard = dict(raw_blackboard) if isinstance(raw_blackboard, Mapping) else {}
            raw_round_summaries = current_state.get("round_summaries")
            round_summaries = (
                [dict(summary) for summary in raw_round_summaries if isinstance(summary, Mapping)]
                if isinstance(raw_round_summaries, list)
                else []
            )
            raw_tool_results = current_state.get("tool_results")
            tool_results = list(raw_tool_results) if isinstance(raw_tool_results, list) else []

            blackboard["round"] = round_number
            round_contributions: dict[str, dict[str, object]] = {}
            explicit_stop = False
            for peer_id in peer_ids:
                peer = self._peers[peer_id]
                peer_prompt = self._peer_prompt_builder(
                    prompt,
                    blackboard,
                    peer_id,
                    round_number,
                )
                peer_dependencies = dict(dependencies)
                peer_dependencies["_networked"] = {
                    "round": round_number,
                    "peer_id": peer_id,
                    "blackboard": _json_ready(blackboard),
                }
                peer_invocation = invoke_delegate(
                    delegate=peer,
                    prompt=peer_prompt,
                    step_context=context,
                    request_id=f"{request_id}:networked:{peer_id}:{round_number}",
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                    dependencies=peer_dependencies,
                )
                peer_result = peer_invocation.result
                tool_results.extend(peer_result.tool_results)
                if not peer_result.success:
                    return {
                        "blackboard": blackboard,
                        "round_summaries": round_summaries,
                        "terminated_reason": "peer_failure",
                        "should_continue": False,
                        "success": False,
                        "failed_peer": peer_id,
                        "peer_output": _json_ready(peer_result.output),
                        "tool_results": tool_results,
                        "last_model_response": peer_result.model_response,
                    }
                contribution = _normalize_peer_contribution(
                    peer_id=peer_id,
                    peer_output=peer_result.output,
                    round_number=round_number,
                )
                round_contributions[peer_id] = contribution
                if contribution.get("stop") is True:
                    explicit_stop = True

            blackboard = self._apply_round_reducer(
                blackboard=blackboard,
                round_number=round_number,
                round_contributions=round_contributions,
            )
            state_hash = _compute_state_hash(blackboard)
            blackboard["state_hash"] = state_hash
            round_summaries.append(
                {
                    "round": round_number,
                    "peer_order": peer_ids,
                    "contributions": _json_ready(round_contributions),
                    "state_hash": state_hash,
                }
            )
            if explicit_stop:
                return {
                    "blackboard": blackboard,
                    "round_summaries": round_summaries,
                    "terminated_reason": "explicit_stop",
                    "should_continue": False,
                    "success": True,
                    "failed_peer": None,
                    "peer_output": {},
                    "tool_results": tool_results,
                    "last_model_response": None,
                }
            if self._check_convergence(
                blackboard=blackboard,
                round_summaries=round_summaries,
            ):
                return {
                    "blackboard": blackboard,
                    "round_summaries": round_summaries,
                    "terminated_reason": "converged",
                    "should_continue": False,
                    "success": True,
                    "failed_peer": None,
                    "peer_output": {},
                    "tool_results": tool_results,
                    "last_model_response": None,
                }
            return {
                "blackboard": blackboard,
                "round_summaries": round_summaries,
                "terminated_reason": "max_rounds_reached",
                "should_continue": True,
                "success": True,
                "failed_peer": None,
                "peer_output": {},
                "tool_results": tool_results,
                "last_model_response": None,
            }

        def _state_reducer(
            state: Mapping[str, object],
            iteration_result: ExecutionResult,
            iteration: int,
        ) -> Mapping[str, object]:
            """Fold one iteration output into accumulated loop state.

            Args:
                state: Current accumulated state.
                iteration_result: Workflow result for one loop iteration.
                iteration: One-based loop iteration index.

            Returns:
                Reduced loop state mapping.
            """
            del iteration
            iteration_step = iteration_result.step_results.get("network_round")
            if iteration_step is None or not getattr(iteration_step, "success", False):
                return dict(state)
            output = getattr(iteration_step, "output", {})
            return dict(output) if isinstance(output, Mapping) else dict(state)

        workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_mode="schema",
            steps=[
                LoopStep(
                    step_id="network_loop",
                    steps=(LogicStep(step_id="network_round", handler=_run_round),),
                    max_iterations=self._max_rounds,
                    initial_state=initial_state,
                    continue_predicate=_continue_predicate,
                    state_reducer=_state_reducer,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                )
            ],
        )
        self.workflow = workflow
        workflow_result = workflow.run(
            {},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:network_workflow",
            dependencies=dependencies,
        )
        final_state = _extract_network_final_state(workflow_result)
        return _build_network_result(
            final_state=final_state,
            workflow_result=workflow_result,
            request_id=request_id,
            dependencies=dependencies,
            peer_ids=peer_ids,
            max_rounds=self._max_rounds,
        )

    def _initial_blackboard(self, prompt: str) -> dict[str, object]:
        """Build initial shared state object.

        Args:
            prompt: Task prompt shared across peers.

        Returns:
            Initial blackboard mapping.
        """
        base = {
            "task": prompt,
            "round": 0,
            "messages": [],
            "proposals": {},
            "decisions": {},
            "history": [],
        }
        base.update(self._initial_state)
        return base

    def _apply_round_reducer(
        self,
        *,
        blackboard: Mapping[str, object],
        round_number: int,
        round_contributions: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        """Default reducer for generic networked pattern.

        Args:
            blackboard: Current shared state.
            round_number: One-based current round number.
            round_contributions: Per-peer round contributions.

        Returns:
            Updated blackboard mapping for the next round.
        """
        next_blackboard = dict(blackboard)
        raw_history = blackboard.get("history")
        history: list[object] = list(raw_history) if isinstance(raw_history, list) else []
        history.append(
            {
                "round": round_number,
                "contributions": _json_ready(round_contributions),
            }
        )
        next_blackboard["history"] = history
        return next_blackboard

    def _check_convergence(
        self,
        *,
        blackboard: Mapping[str, object],
        round_summaries: Sequence[Mapping[str, object]],
    ) -> bool:
        """Return whether the networked pattern converged.

        Args:
            blackboard: Current shared state.
            round_summaries: Ordered round summary payloads.

        Returns:
            ``True`` when convergence criteria are met.
        """
        del blackboard, round_summaries
        return False


class BlackboardPattern(NetworkedPattern):
    """Networked pattern with explicit blackboard reducer semantics."""

    def __init__(
        self,
        *,
        peers: Mapping[str, WorkflowDelegate],
        max_rounds: int = 6,
        stability_rounds: int = 2,
        initial_state: Mapping[str, object] | None = None,
        peer_prompt_builder: (Callable[[str, Mapping[str, object], str, int], str] | None) = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize blackboard specialization with convergence controls.

        Args:
            peers: Peer delegates participating in rounds.
            max_rounds: Maximum rounds before termination.
            stability_rounds: Number of unchanged state hashes required to
                declare convergence.
            initial_state: Optional initial blackboard override mapping.
            peer_prompt_builder: Optional peer prompt builder callback.
            tracer: Optional tracer dependency.

        Raises:
            ValueError: Raised when ``stability_rounds`` is less than one.
        """
        if stability_rounds < 1:
            raise ValueError("stability_rounds must be >= 1.")
        super().__init__(
            peers=peers,
            max_rounds=max_rounds,
            initial_state=initial_state,
            peer_prompt_builder=peer_prompt_builder,
            tracer=tracer,
        )
        self._stability_rounds = stability_rounds

    def _apply_round_reducer(
        self,
        *,
        blackboard: Mapping[str, object],
        round_number: int,
        round_contributions: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        """Merge peer contributions into blackboard channels.

        Args:
            blackboard: Current shared blackboard state.
            round_number: One-based current round number.
            round_contributions: Per-peer round contributions.

        Returns:
            Updated blackboard mapping.
        """
        next_blackboard = dict(blackboard)

        raw_messages_value = blackboard.get("messages")
        messages: list[object] = (
            list(raw_messages_value) if isinstance(raw_messages_value, list) else []
        )

        raw_proposals_value = blackboard.get("proposals")
        proposals: dict[str, object]
        if isinstance(raw_proposals_value, Mapping):
            proposals = {str(key): value for key, value in raw_proposals_value.items()}
        else:
            proposals = {}

        raw_decisions_value = blackboard.get("decisions")
        decisions: dict[str, object]
        if isinstance(raw_decisions_value, Mapping):
            decisions = {str(key): value for key, value in raw_decisions_value.items()}
        else:
            decisions = {}

        raw_history_value = blackboard.get("history")
        history: list[object] = (
            list(raw_history_value) if isinstance(raw_history_value, list) else []
        )

        for peer_id in sorted(round_contributions):
            contribution = round_contributions[peer_id]
            raw_messages = contribution.get("messages")
            if isinstance(raw_messages, list):
                for message in raw_messages:
                    messages.append(
                        {
                            "round": round_number,
                            "peer_id": peer_id,
                            "message": _json_ready(message),
                        }
                    )
            raw_proposals = contribution.get("proposals")
            if isinstance(raw_proposals, Mapping):
                proposals[peer_id] = _json_ready(dict(raw_proposals))
            raw_decisions = contribution.get("decisions")
            if isinstance(raw_decisions, Mapping):
                decisions[peer_id] = _json_ready(dict(raw_decisions))

        next_blackboard["task"] = str(blackboard.get("task", ""))
        next_blackboard["round"] = round_number
        next_blackboard["messages"] = messages
        next_blackboard["proposals"] = proposals
        next_blackboard["decisions"] = decisions
        next_blackboard["history"] = [
            *history,
            {
                "round": round_number,
                "contributions": _json_ready(round_contributions),
            },
        ]
        next_blackboard["state_hash"] = _compute_state_hash(next_blackboard)
        return next_blackboard

    def _check_convergence(
        self,
        *,
        blackboard: Mapping[str, object],
        round_summaries: Sequence[Mapping[str, object]],
    ) -> bool:
        """Return ``True`` after stable state hash repeats for configured rounds.

        Args:
            blackboard: Current shared blackboard state.
            round_summaries: Ordered round summaries.

        Returns:
            ``True`` when recent state hashes are identical.
        """
        del blackboard
        if len(round_summaries) < self._stability_rounds:
            return False

        recent_hashes: list[str] = []
        for summary in round_summaries[-self._stability_rounds :]:
            state_hash = summary.get("state_hash")
            if not isinstance(state_hash, str) or not state_hash:
                return False
            recent_hashes.append(state_hash)
        return len(set(recent_hashes)) == 1


def _extract_network_final_state(workflow_result: ExecutionResult) -> dict[str, object]:
    """Extract normalized final loop state from network workflow output.

    Args:
        workflow_result: Workflow execution result produced by ``NetworkedPattern``.

    Returns:
        Final loop-state mapping extracted from ``network_loop`` output.

    Raises:
        RuntimeError: Raised when ``network_loop`` result is missing.
    """
    loop_step_result = workflow_result.step_results.get("network_loop")
    if loop_step_result is None:
        raise RuntimeError("Network loop step result is missing.")
    loop_output = loop_step_result.output
    final_state_raw = loop_output.get("final_state")
    return dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}


def _build_network_result(
    *,
    final_state: Mapping[str, object],
    workflow_result: ExecutionResult,
    request_id: str,
    dependencies: Mapping[str, object],
    peer_ids: Sequence[str],
    max_rounds: int,
) -> ExecutionResult:
    """Build final ``ExecutionResult`` payload for networked patterns.

    Args:
        final_state: Final network loop state.
        workflow_result: Workflow execution result.
        request_id: Resolved request id.
        dependencies: Dependency mapping used for the run.
        peer_ids: Ordered peer identifiers.
        max_rounds: Configured max rounds.

    Returns:
        Final normalized execution result.
    """
    blackboard_raw = final_state.get("blackboard")
    blackboard = dict(blackboard_raw) if isinstance(blackboard_raw, Mapping) else {}
    raw_round_summaries = final_state.get("round_summaries")
    round_summaries = (
        [dict(summary) for summary in raw_round_summaries if isinstance(summary, Mapping)]
        if isinstance(raw_round_summaries, list)
        else []
    )
    terminated_reason = str(final_state.get("terminated_reason", "max_rounds_reached"))
    success = bool(final_state.get("success", True)) and workflow_result.success
    failed_peer = final_state.get("failed_peer")
    peer_output = final_state.get("peer_output", {})
    raw_tool_results = final_state.get("tool_results")
    tool_results = list(raw_tool_results) if isinstance(raw_tool_results, list) else []
    maybe_model_response = final_state.get("last_model_response")
    model_response = maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None

    output: dict[str, object] = {
        "blackboard": _json_ready(blackboard),
        "round_summaries": round_summaries,
        "terminated_reason": terminated_reason,
        "rounds_executed": len(round_summaries),
        "final_output": {
            "blackboard": _json_ready(blackboard),
            "rounds_executed": len(round_summaries),
            "terminated_reason": terminated_reason,
        },
        "workflow": workflow_result.asdict(),
        "artifacts": workflow_result.output.get("artifacts", []),
    }
    if isinstance(failed_peer, str) and failed_peer:
        output["failed_peer"] = failed_peer
        output["peer_output"] = _json_ready(peer_output)

    return ExecutionResult(
        output=output,
        success=success,
        tool_results=tool_results,
        model_response=model_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            "peer_order": list(peer_ids),
            "max_rounds": max_rounds,
        },
    )


def _default_peer_prompt_builder(
    task_prompt: str,
    blackboard: Mapping[str, object],
    peer_id: str,
    round_number: int,
) -> str:
    """Build default peer prompt containing shared blackboard state.

    Args:
        task_prompt: Shared task prompt.
        blackboard: Current blackboard state.
        peer_id: Peer identifier.
        round_number: One-based round number.

    Returns:
        JSON prompt payload for one peer.
    """
    payload = {
        "task": task_prompt,
        "peer_id": peer_id,
        "round": round_number,
        "blackboard": _json_ready(blackboard),
        "instructions": [
            "Contribute as a peer in a networked blackboard system.",
            "Return JSON with optional keys: messages, proposals, decisions, stop.",
        ],
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _normalize_peer_contribution(
    *,
    peer_id: str,
    peer_output: Mapping[str, object],
    round_number: int,
) -> dict[str, object]:
    """Normalize arbitrary peer output into blackboard contribution channels.

    Args:
        peer_id: Peer identifier.
        peer_output: Raw peer output mapping.
        round_number: One-based round number.

    Returns:
        Normalized per-peer contribution mapping.
    """
    contribution: dict[str, object] = {
        "peer_id": peer_id,
        "round": round_number,
        "messages": [],
        "proposals": {},
        "decisions": {},
        "stop": False,
    }

    raw_messages = peer_output.get("messages")
    if isinstance(raw_messages, list):
        contribution["messages"] = list(raw_messages)
    elif isinstance(peer_output.get("message"), str):
        contribution["messages"] = [str(peer_output["message"])]
    else:
        contribution["messages"] = [json.dumps(_json_ready(peer_output), ensure_ascii=True)]

    raw_proposals = peer_output.get("proposals")
    if isinstance(raw_proposals, Mapping):
        contribution["proposals"] = dict(raw_proposals)

    raw_decisions = peer_output.get("decisions")
    if isinstance(raw_decisions, Mapping):
        contribution["decisions"] = dict(raw_decisions)

    raw_stop = peer_output.get("stop")
    if isinstance(raw_stop, bool):
        contribution["stop"] = raw_stop

    return contribution


def _compute_state_hash(payload: Mapping[str, object]) -> str:
    """Compute deterministic hash for blackboard convergence checks.

    Args:
        payload: Blackboard payload to hash.

    Returns:
        Deterministic state hash.
    """
    stable_payload = {
        "messages": _json_ready(payload.get("messages", [])),
        "proposals": _json_ready(payload.get("proposals", {})),
        "decisions": _json_ready(payload.get("decisions", {})),
    }
    serialized = json.dumps(stable_payload, ensure_ascii=True, sort_keys=True)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _safe_int(value: object) -> int:
    """Convert values to int with deterministic fallback to zero.

    Args:
        value: Value to normalize into an integer.

    Returns:
        Integer representation, or ``0`` when conversion fails.
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
    """Recursively convert mappings/sequences into JSON-serializable primitives.

    Args:
        value: Arbitrary value.

    Returns:
        JSON-serializable representation.
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


__all__ = ["BlackboardPattern", "NetworkedPattern"]
