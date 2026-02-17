"""Tests for networked and blackboard workflow patterns."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from design_research_agents.contracts.agent import Agent, AgentResult
from design_research_agents.workflow import BlackboardPattern, NetworkedPattern


class _RecordingPeerAgent(Agent):
    """Peer agent that records invocation order and emits fixed payload."""

    def __init__(self, *, peer_id: str, recorder: list[str], payload: Mapping[str, object]) -> None:
        self._peer_id = peer_id
        self._recorder = recorder
        self._payload = dict(payload)

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del prompt, request_id, dependencies
        self._recorder.append(self._peer_id)
        return AgentResult(
            output=dict(self._payload),
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"peer_id": self._peer_id},
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[object]:
        del prompt, request_id, dependencies
        raise NotImplementedError


class _FailingPeerAgent(Agent):
    """Peer agent that deterministically fails."""

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del prompt, request_id, dependencies
        return AgentResult(
            output={"error": "peer failed"},
            success=False,
            tool_results=[],
            model_response=None,
            metadata={"peer": "failing"},
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[object]:
        del prompt, request_id, dependencies
        raise NotImplementedError


def test_networked_pattern_uses_deterministic_sorted_peer_order_each_round() -> None:
    call_order: list[str] = []
    pattern = NetworkedPattern(
        peers={
            "peer_b": _RecordingPeerAgent(
                peer_id="peer_b",
                recorder=call_order,
                payload={"messages": ["b"]},
            ),
            "peer_a": _RecordingPeerAgent(
                peer_id="peer_a",
                recorder=call_order,
                payload={"messages": ["a"]},
            ),
        },
        max_rounds=2,
    )

    result = pattern.run("Coordinate this task.")

    assert result.success
    assert result.output["terminated_reason"] == "max_rounds_reached"
    assert result.output["rounds_executed"] == 2
    assert call_order == ["peer_a", "peer_b", "peer_a", "peer_b"]


def test_blackboard_pattern_converges_after_stable_state_hash_rounds() -> None:
    pattern = BlackboardPattern(
        peers={
            "peer_1": _RecordingPeerAgent(
                peer_id="peer_1",
                recorder=[],
                payload={"messages": [], "proposals": {}, "decisions": {}},
            ),
            "peer_2": _RecordingPeerAgent(
                peer_id="peer_2",
                recorder=[],
                payload={"messages": [], "proposals": {}, "decisions": {}},
            ),
        },
        max_rounds=6,
        stability_rounds=2,
    )

    result = pattern.run("Need stable convergence.")

    assert result.success
    assert result.output["terminated_reason"] == "converged"
    assert result.output["rounds_executed"] == 2
    round_summaries = result.output["round_summaries"]
    assert round_summaries[0]["state_hash"] == round_summaries[1]["state_hash"]


def test_blackboard_pattern_honors_explicit_stop_signal() -> None:
    pattern = BlackboardPattern(
        peers={
            "peer_a": _RecordingPeerAgent(
                peer_id="peer_a",
                recorder=[],
                payload={"messages": ["done"], "stop": True},
            )
        },
        max_rounds=5,
    )

    result = pattern.run("Stop after one round.")

    assert result.success
    assert result.output["terminated_reason"] == "explicit_stop"
    assert result.output["rounds_executed"] == 1


def test_networked_pattern_returns_peer_failure_termination() -> None:
    pattern = NetworkedPattern(
        peers={
            "peer_a": _RecordingPeerAgent(
                peer_id="peer_a",
                recorder=[],
                payload={"messages": ["ok"]},
            ),
            "peer_b": _FailingPeerAgent(),
        },
        max_rounds=3,
    )

    result = pattern.run("One peer fails.")

    assert not result.success
    assert result.output["terminated_reason"] == "peer_failure"
    assert result.output["failed_peer"] == "peer_b"
