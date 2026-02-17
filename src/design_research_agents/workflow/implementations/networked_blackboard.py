"""Networked and blackboard workflow patterns without central orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from hashlib import sha256

from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    resolve_request_id,
)
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.tracing import Tracer, finish_trace_run, start_trace_run


class NetworkedPattern(Agent):
    """Round-based peer coordination pattern with deterministic peer ordering."""

    def __init__(
        self,
        *,
        peers: Mapping[str, Agent],
        max_rounds: int = 4,
        initial_state: Mapping[str, object] | None = None,
        peer_prompt_builder: Callable[[str, Mapping[str, object], str, int], str] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize peer-only networked orchestration.

        Args:
            peers: Mapping of peer ids to agent delegates.
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

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
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

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Stream wrapper around ``run`` with one summary delta.

        Args:
            prompt: Task prompt shared across peers.
            request_id: Optional request id.
            dependencies: Optional dependency mapping.

        Yields:
            One delta event followed by one completed event.
        """
        result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        yield AgentStreamEvent(
            kind="delta",
            delta_text=json.dumps(result.output, ensure_ascii=True, sort_keys=True),
        )
        yield AgentStreamEvent(kind="completed", result=result)

    def _run_network(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> AgentResult:
        """Execute the core peer coordination loop.

        Args:
            prompt: Task prompt shared across peers.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency mapping.

        Returns:
            Final coordination result payload.
        """
        blackboard = self._initial_blackboard(prompt)
        round_summaries: list[dict[str, object]] = []
        terminated_reason = "max_rounds_reached"
        peer_ids = sorted(self._peers)

        for round_number in range(1, self._max_rounds + 1):
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
                peer_result = peer.run(
                    peer_prompt,
                    request_id=f"{request_id}:networked:{peer_id}:{round_number}",
                    dependencies=peer_dependencies,
                )
                if not peer_result.success:
                    terminated_reason = "peer_failure"
                    return AgentResult(
                        output={
                            "blackboard": _json_ready(blackboard),
                            "round_summaries": round_summaries,
                            "terminated_reason": terminated_reason,
                            "rounds_executed": len(round_summaries),
                            "failed_peer": peer_id,
                            "peer_output": _json_ready(peer_result.output),
                        },
                        success=False,
                        tool_results=list(peer_result.tool_results),
                        model_response=peer_result.model_response,
                        metadata={
                            "request_id": request_id,
                            "dependency_keys": sorted(dependencies.keys()),
                            "peer_order": peer_ids,
                            "failed_peer": peer_id,
                        },
                    )

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

            round_summary = {
                "round": round_number,
                "peer_order": peer_ids,
                "contributions": _json_ready(round_contributions),
                "state_hash": state_hash,
            }
            round_summaries.append(round_summary)

            if explicit_stop:
                terminated_reason = "explicit_stop"
                break

            if self._check_convergence(
                blackboard=blackboard,
                round_summaries=round_summaries,
            ):
                terminated_reason = "converged"
                break

        return AgentResult(
            output={
                "blackboard": _json_ready(blackboard),
                "round_summaries": round_summaries,
                "terminated_reason": terminated_reason,
                "rounds_executed": len(round_summaries),
            },
            success=True,
            tool_results=[],
            model_response=None,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "peer_order": peer_ids,
                "max_rounds": self._max_rounds,
            },
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
        peers: Mapping[str, Agent],
        max_rounds: int = 6,
        stability_rounds: int = 2,
        initial_state: Mapping[str, object] | None = None,
        peer_prompt_builder: Callable[[str, Mapping[str, object], str, int], str] | None = None,
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
