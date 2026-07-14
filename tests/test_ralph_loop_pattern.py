"""Tests for RalphLoopPattern dynamic-role loop behavior."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._implementations._patterns import _ralph_loop_pattern as ralph_impl
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


class _PromptRecordingRoleAgent:
    def __init__(self, *, output: Mapping[str, object]) -> None:
        self._output = dict(output)
        self.prompts: list[str] = []

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del request_id, dependencies
        self.prompts.append(prompt)
        return ExecutionResult(output=dict(self._output), success=True, tool_results=[], model_response=None)


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

    with pytest.raises(ValueError, match="non-empty role_id"):
        RalphLoopPattern(
            roles=(RalphLoopPattern.RoleSpec(role_id=" ", delegate=_SequenceRoleAgent([{}])),),
            evaluator_role_id=" ",
        )

    with pytest.raises(ValueError, match="selection_strategy"):
        RalphLoopPattern(
            roles=(RalphLoopPattern.RoleSpec(role_id="evaluator", delegate=_SequenceRoleAgent([{"score": 1.0}])),),
            evaluator_role_id="evaluator",
            loop_config=RalphLoopPattern.LoopConfig(selection_strategy="invalid"),  # type: ignore[arg-type]
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


def test_ralph_loop_pattern_passes_current_round_outputs_to_later_roles() -> None:
    proposer = _PromptRecordingRoleAgent(output={"proposal": "draft v1"})
    critic = _PromptRecordingRoleAgent(output={"risks": ["needs sealing"]})
    synthesizer = _PromptRecordingRoleAgent(output={"synthesis": "draft v1 with sealing"})
    evaluator = _PromptRecordingRoleAgent(output={"score": 0.9})

    pattern = RalphLoopPattern(
        roles=(
            RalphLoopPattern.RoleSpec(
                role_id="proposer",
                delegate=proposer,
                prompt_template="Selected={selected_output_json} Prior={prior_role_outputs_json}",
            ),
            RalphLoopPattern.RoleSpec(
                role_id="critic",
                delegate=critic,
                prompt_template="Prior={prior_role_outputs_json}",
            ),
            RalphLoopPattern.RoleSpec(
                role_id="synthesizer",
                delegate=synthesizer,
                prompt_template="Prior={prior_role_outputs_json}",
            ),
            RalphLoopPattern.RoleSpec(
                role_id="evaluator",
                delegate=evaluator,
                prompt_template="Selected={selected_output_json} Prior={prior_role_outputs_json}",
            ),
        ),
        evaluator_role_id="evaluator",
        loop_config=RalphLoopPattern.LoopConfig(max_iterations=1, consensus_threshold=0.8),
    )

    result = pattern.run("Refine this concept.")

    assert result.success
    assert proposer.prompts
    assert critic.prompts
    assert synthesizer.prompts
    assert evaluator.prompts
    assert '"proposal": "draft v1"' in critic.prompts[0]
    assert '"proposal": "draft v1"' in synthesizer.prompts[0]
    assert '"risks": [' in synthesizer.prompts[0]
    assert '"synthesis": "draft v1 with sealing"' in evaluator.prompts[0]


def test_ralph_internal_role_and_evaluator_helpers_report_missing_results() -> None:
    pattern = RalphLoopPattern(
        roles=(
            RalphLoopPattern.RoleSpec(role_id="proposer", delegate=_SequenceRoleAgent([{}])),
            RalphLoopPattern.RoleSpec(role_id="evaluator", delegate=_SequenceRoleAgent([{}])),
        ),
        evaluator_role_id="evaluator",
    )

    assert pattern._resolve_iteration({}) == 1
    assert pattern._find_role_failure({}) == ("proposer", "Role result missing.")
    assert pattern._find_role_failure({"proposer": {"success": False}}) == ("proposer", "Role call failed.")
    assert pattern._extract_evaluator_score({}) == (0.0, "Evaluator role result is missing.")
    assert pattern._extract_evaluator_score({"evaluator": {"output": "invalid"}}) == (
        0.0,
        "Evaluator role output is missing.",
    )

    with pytest.raises(RuntimeError, match="step result is missing"):
        pattern._finalize_result(
            workflow_result=ExecutionResult(success=False),
            request_id="request",
            dependencies={},
        )


def test_ralph_synthesis_and_compaction_helpers_cover_fallback_shapes() -> None:
    roles = (
        RalphLoopPattern.RoleSpec(role_id="proposer", delegate=object()),
        RalphLoopPattern.RoleSpec(role_id="evaluator", delegate=object()),
    )
    assert ralph_impl._resolve_synthesized_output(
        {"proposer": "invalid", "evaluator": {"output": {"feedback": "only"}}},  # type: ignore[dict-item]
        ordered_roles=roles,
        evaluator_role_id="evaluator",
    ) == {"feedback": "only"}
    assert (
        ralph_impl._resolve_synthesized_output(
            {},
            ordered_roles=roles,
            evaluator_role_id="evaluator",
        )
        == {}
    )
    assert ralph_impl._compact_role_results(
        {
            "invalid": "ignored",
            "valid": {"success": 1, "error": None, "output": {"final_output": '{"value": 2}'}},
        }
    ) == {"valid": {"success": True, "error": None, "output": {"value": 2}}}

    assert ralph_impl._compact_role_output({}) == {}
    assert ralph_impl._compact_role_output({"final_output": 3}) == {"value": 3}
    assert ralph_impl._compact_role_output({"model_text": " plain text "}) == {"text": "plain text"}
    assert ralph_impl._compact_role_output(
        {
            "workflow": {"large": True},
            "artifacts": ["ignored"],
            "mapping": {"a": 1},
            "items": [1, 2],
            "label": " value ",
            "count": 2,
            "none": None,
        }
    ) == {"mapping": {"a": 1}, "items": [1, 2], "label": "value", "count": 2}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"a": object()}, {"a": "<object>"}),
        (" ", {}),
        ("[1, 2]", {"text": "[1, 2]"}),
        (True, {"value": True}),
        (None, {}),
    ],
)
def test_ralph_compact_value_variants(value: object, expected: dict[str, object]) -> None:
    result = ralph_impl._mapping_from_compact_value(value)
    if value and isinstance(value, Mapping):
        assert result["a"].startswith("<object object at")
    else:
        assert result == expected


def test_ralph_score_error_and_scalar_helpers_cover_invalid_inputs() -> None:
    assert (
        ralph_impl._extract_execution_error(
            result=ExecutionResult(success=False, output={"error": " explicit "}),
            fallback="fallback",
        )
        == "explicit"
    )
    assert (
        ralph_impl._extract_execution_error(
            result=SimpleNamespace(error=None, output={"error": " output "}),  # type: ignore[arg-type]
            fallback="fallback",
        )
        == "output"
    )
    assert ralph_impl._extract_execution_error(result=ExecutionResult(success=False), fallback="fallback") == "fallback"

    assert ralph_impl._extract_score({"score": 3}) == 1.0
    assert ralph_impl._extract_score({"model_text": '{"score": -1}'}) == 0.0
    assert ralph_impl._extract_score({"model_text": "invalid"}) is None
    assert ralph_impl._extract_score({"model_text": '{"score": "high"}'}) is None
    assert ralph_impl._extract_score({}) is None

    assert ralph_impl._safe_float(True) == 1.0
    assert ralph_impl._safe_float(2) == 2.0
    assert ralph_impl._safe_float(" 2.5 ") == 2.5
    assert ralph_impl._safe_float("invalid") == 0.0
    assert ralph_impl._safe_float(object()) == 0.0
    assert ralph_impl._safe_int(True) == 1
    assert ralph_impl._safe_int(2) == 2
    assert ralph_impl._safe_int(2.9) == 2
    assert ralph_impl._safe_int(" 3 ") == 3
    assert ralph_impl._safe_int("invalid") == 0
    assert ralph_impl._safe_int(object()) == 0
    assert ralph_impl._as_dict({"a": 1}) == {"a": 1}
    assert ralph_impl._as_dict("invalid") == {}
    assert ralph_impl._as_list([1]) == [1]
    assert ralph_impl._as_list((1,)) == []
    assert ralph_impl._json_ready({"a": (1, object())})["a"][0] == 1
    assert ralph_impl._json_ready([1, object()])[0] == 1
