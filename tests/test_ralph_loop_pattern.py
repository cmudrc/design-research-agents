"""Tests for RalphLoopPattern dynamic-role loop behavior."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents.patterns import RalphLoopPattern


class _SequenceRoleAgent:
    def __init__(self, outputs: list[Mapping[str, object]]) -> None:
        self._outputs = [dict(output) for output in outputs]
        self._index = 0

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        output = {} if not self._outputs else self._outputs[min(self._index, len(self._outputs) - 1)]
        self._index += 1
        return ExecutionResult(output=dict(output), success=True, tool_results=[], model_response=None)


def test_ralph_loop_pattern_validates_role_and_config_inputs() -> None:
    with pytest.raises(ValueError, match="at least one role"):
        RalphLoopPattern(
            roles=(),
            evaluator_role_id="evaluator",
        )

    with pytest.raises(ValueError, match="unique role_id"):
        RalphLoopPattern(
            roles=(
                RalphLoopPattern.RoleSpec(role_id="role", delegate=_SequenceRoleAgent([{}])),
                RalphLoopPattern.RoleSpec(role_id="role", delegate=_SequenceRoleAgent([{}])),
            ),
            evaluator_role_id="role",
        )

    with pytest.raises(ValueError, match="evaluator_role_id"):
        RalphLoopPattern(
            roles=(RalphLoopPattern.RoleSpec(role_id="proposer", delegate=_SequenceRoleAgent([{}])),),
            evaluator_role_id="missing",
        )

    with pytest.raises(ValueError, match="max_iterations"):
        RalphLoopPattern(
            roles=(RalphLoopPattern.RoleSpec(role_id="evaluator", delegate=_SequenceRoleAgent([{"score": 1.0}])),),
            evaluator_role_id="evaluator",
            loop_config=RalphLoopPattern.LoopConfig(max_iterations=0),
        )

    with pytest.raises(ValueError, match="consensus_threshold"):
        RalphLoopPattern(
            roles=(RalphLoopPattern.RoleSpec(role_id="evaluator", delegate=_SequenceRoleAgent([{"score": 1.0}])),),
            evaluator_role_id="evaluator",
            loop_config=RalphLoopPattern.LoopConfig(consensus_threshold=1.1),
        )


def test_ralph_loop_pattern_stops_early_when_threshold_reached() -> None:
    pattern = RalphLoopPattern(
        roles=(
            RalphLoopPattern.RoleSpec(
                role_id="proposer",
                delegate=_SequenceRoleAgent(
                    outputs=[
                        {"proposal": "draft v1"},
                        {"proposal": "draft v2"},
                    ]
                ),
            ),
            RalphLoopPattern.RoleSpec(
                role_id="evaluator",
                delegate=_SequenceRoleAgent(outputs=[{"score": 0.4}, {"score": 0.85}]),
            ),
        ),
        evaluator_role_id="evaluator",
        loop_config=RalphLoopPattern.LoopConfig(
            max_iterations=4,
            consensus_threshold=0.8,
            selection_strategy="best_score",
        ),
    )

    result = pattern.run("Refine this concept.")

    assert result.success
    assert result.output["terminated_reason"] == "consensus_reached"
    assert result.output["details"]["consensus_score"] == 0.85
    assert result.output["details"]["best_score"] == 0.85
    assert result.output["details"]["best_iteration"] == 2
    assert len(result.output["details"]["iteration_history"]) == 2
    first_roles = result.output["details"]["iteration_history"][0]["roles"]
    assert first_roles["proposer"]["error"] is None
    assert first_roles["evaluator"]["error"] is None


def test_ralph_loop_pattern_uses_max_iteration_fallback_and_latest_selection() -> None:
    pattern = RalphLoopPattern(
        roles=(
            RalphLoopPattern.RoleSpec(
                role_id="proposer",
                delegate=_SequenceRoleAgent(
                    outputs=[
                        {"proposal": "draft v1"},
                        {"proposal": "draft v2"},
                        {"proposal": "draft v3"},
                    ]
                ),
            ),
            RalphLoopPattern.RoleSpec(
                role_id="evaluator",
                delegate=_SequenceRoleAgent(outputs=[{"score": 0.2}, {"score": 0.3}, {"score": 0.4}]),
            ),
        ),
        evaluator_role_id="evaluator",
        loop_config=RalphLoopPattern.LoopConfig(
            max_iterations=3,
            consensus_threshold=0.8,
            selection_strategy="latest",
        ),
    )

    result = pattern.run("Refine this concept.")

    assert result.success
    assert result.output["terminated_reason"] == "max_iterations_reached"
    assert result.output["final_output"]["proposal"] == "draft v3"
    assert result.output["details"]["best_score"] == 0.4
    assert len(result.output["details"]["iteration_history"]) == 3


def test_ralph_loop_pattern_reports_unknown_template_variables() -> None:
    pattern = RalphLoopPattern(
        roles=(
            RalphLoopPattern.RoleSpec(
                role_id="proposer",
                delegate=_SequenceRoleAgent(outputs=[{"proposal": "draft"}]),
                prompt_template="Task={task} Missing={not_a_known_key}",
            ),
            RalphLoopPattern.RoleSpec(
                role_id="evaluator",
                delegate=_SequenceRoleAgent(outputs=[{"score": 1.0}]),
            ),
        ),
        evaluator_role_id="evaluator",
        loop_config=RalphLoopPattern.LoopConfig(max_iterations=1),
    )

    result = pattern.run("Refine this concept.")

    assert not result.success
    assert result.output["terminated_reason"] == "role_failure"
    assert "unknown variable" in str(result.error)
    history = result.output["details"]["iteration_history"]
    assert len(history) == 1
    first_roles = history[0]["roles"]
    assert "unknown variable" in str(first_roles["proposer"]["error"])


def test_ralph_loop_pattern_requires_evaluator_score_payload() -> None:
    pattern = RalphLoopPattern(
        roles=(
            RalphLoopPattern.RoleSpec(
                role_id="proposer",
                delegate=_SequenceRoleAgent(outputs=[{"proposal": "draft v1"}]),
            ),
            RalphLoopPattern.RoleSpec(
                role_id="evaluator",
                delegate=_SequenceRoleAgent(outputs=[{"feedback": "no score provided"}]),
            ),
        ),
        evaluator_role_id="evaluator",
        loop_config=RalphLoopPattern.LoopConfig(max_iterations=2),
    )

    result = pattern.run("Refine this concept.")

    assert not result.success
    assert result.output["terminated_reason"] == "role_failure"
    assert "score" in str(result.error)
    history = result.output["details"]["iteration_history"]
    assert len(history) == 1
    assert history[0]["failed_role"] == "evaluator"
