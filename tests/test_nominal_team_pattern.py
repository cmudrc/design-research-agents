"""Tests for nominal-team fan-out and best-of-N selection."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._implementations._patterns import (
    _nominal_team_pattern as nominal_team_impl,
)
from design_research_agents.patterns import NominalTeamPattern


class _StaticTeamAgent(Delegate):
    def __init__(self, *, output: Mapping[str, object], success: bool = True) -> None:
        self._output = dict(output)
        self._success = success

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return ExecutionResult(
            output=dict(self._output),
            success=self._success,
            tool_results=[],
            model_response=None,
        )


class _StaticEvaluatorAgent(Delegate):
    def __init__(self, *, output: Mapping[str, object], success: bool = True) -> None:
        self._output = dict(output)
        self._success = success

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return ExecutionResult(
            output=dict(self._output),
            success=self._success,
            tool_results=[],
            model_response=None,
        )


class _CapturingEvaluatorAgent(Delegate):
    def __init__(self, *, output: Mapping[str, object] | None = None) -> None:
        self.prompts: list[str] = []
        self._output = dict(output or {"best_index": 0})

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


class _ExplodingEvaluatorAgent(Delegate):
    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        raise RuntimeError("evaluator exploded")


def test_nominal_team_pattern_validates_member_inputs() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        NominalTeamPattern(team_members=(), evaluator_delegate=lambda _context: {"best_index": 0})

    with pytest.raises(ValueError, match="unique member_id"):
        NominalTeamPattern(
            team_members=(
                NominalTeamPattern.MemberSpec(
                    member_id="designer",
                    delegate=_StaticTeamAgent(output={"concept": "a"}),
                ),
                NominalTeamPattern.MemberSpec(
                    member_id="designer",
                    delegate=_StaticTeamAgent(output={"concept": "b"}),
                ),
            ),
            evaluator_delegate=lambda _context: {"best_index": 0},
        )

    with pytest.raises(ValueError, match="unknown variable"):
        NominalTeamPattern(
            team_members=(
                NominalTeamPattern.MemberSpec(
                    member_id="designer",
                    delegate=_StaticTeamAgent(output={"concept": "a"}),
                    prompt_template="Task={task} Persona={persona}",
                ),
            ),
            evaluator_delegate=lambda _context: {"best_index": 0},
        )


def test_nominal_team_pattern_selects_best_member_and_records_scores() -> None:
    pattern = NominalTeamPattern(
        team_members=(
            NominalTeamPattern.MemberSpec(
                member_id="repairability",
                delegate=_StaticTeamAgent(output={"concept": "modular hatch", "risk": "extra fasteners"}),
            ),
            NominalTeamPattern.MemberSpec(
                member_id="reliability",
                delegate=_StaticTeamAgent(output={"concept": "sealed latch", "risk": "tighter tolerances"}),
            ),
        ),
        evaluator_delegate=lambda _context: {
            "best_member_id": "reliability",
            "scores": {"repairability": 0.55, "reliability": 0.82},
            "rationale": "Higher reliability score under the active rubric.",
        },
    )

    result = pattern.run("Choose the strongest enclosure concept.")

    assert result.success
    assert pattern.workflow is not None
    assert result.output["final_output"]["concept"] == "sealed latch"
    assert result.output["details"]["selected_member_id"] == "reliability"
    assert result.output["details"]["candidate_count"] == 2
    assert result.output["details"]["scores"]["reliability"] == 0.82
    assert result.output["details"]["selection_score"] == 0.82
    assert [entry["candidate_index"] for entry in result.output["details"]["candidate_results"]] == [0, 1]


def test_nominal_team_pattern_supports_best_index_with_partial_member_failures() -> None:
    pattern = NominalTeamPattern(
        team_members=(
            NominalTeamPattern.MemberSpec(
                member_id="designer_a",
                delegate=_StaticTeamAgent(output={"model_text": '{"concept":"concept-a","novelty":0.4}'}),
            ),
            NominalTeamPattern.MemberSpec(
                member_id="designer_b",
                delegate=_StaticTeamAgent(output={"error": "delegate failed"}, success=False),
            ),
            NominalTeamPattern.MemberSpec(
                member_id="designer_c",
                delegate=_StaticTeamAgent(output={"model_text": "Plain-text concept from designer C"}),
            ),
        ),
        evaluator_delegate=_StaticEvaluatorAgent(output={"best_index": 1, "scores": [0.4, 0.9]}),
    )

    result = pattern.run("Generate three concept directions.")

    assert result.success
    assert result.output["final_output"]["text"] == "Plain-text concept from designer C"
    assert result.output["details"]["selected_member_id"] == "designer_c"
    assert result.output["details"]["selected_candidate_index"] == 1
    assert result.output["details"]["selected_team_index"] == 2
    member_results = result.output["details"]["candidate_results"]
    assert member_results[1]["success"] is False
    assert member_results[1]["candidate_ready"] is False
    assert member_results[1]["error"] == "delegate failed"


def test_nominal_team_pattern_reports_evaluation_failure() -> None:
    pattern = NominalTeamPattern(
        team_members=(
            NominalTeamPattern.MemberSpec(
                member_id="designer_a",
                delegate=_StaticTeamAgent(output={"concept": "concept-a"}),
            ),
            NominalTeamPattern.MemberSpec(
                member_id="designer_b",
                delegate=_StaticTeamAgent(output={"concept": "concept-b"}),
            ),
        ),
        evaluator_delegate=_StaticEvaluatorAgent(output={"error": "rubric unavailable"}, success=False),
    )

    result = pattern.run("Select the best concept.")

    assert not result.success
    assert result.output["terminated_reason"] == "evaluation_failure"
    assert result.error == "rubric unavailable"
    assert result.output["details"]["candidate_count"] == 2


def test_nominal_team_pattern_reports_evaluator_exceptions_as_evaluation_failures() -> None:
    pattern = NominalTeamPattern(
        team_members=(
            NominalTeamPattern.MemberSpec(
                member_id="designer_a",
                delegate=_StaticTeamAgent(output={"concept": "concept-a"}),
            ),
        ),
        evaluator_delegate=_ExplodingEvaluatorAgent(),
    )

    result = pattern.run("Select the best concept.")

    assert not result.success
    assert result.output["terminated_reason"] == "evaluation_failure"
    assert result.error == "evaluator exploded"


def test_nominal_team_pattern_compacts_evaluator_prompt_from_workflow_first_member_outputs() -> None:
    evaluator = _CapturingEvaluatorAgent()
    pattern = NominalTeamPattern(
        team_members=(
            NominalTeamPattern.MemberSpec(
                member_id="designer_a",
                delegate=_StaticTeamAgent(
                    output={
                        "model_text": '{"concept":"concept-a","risk":"extra fasteners"}',
                        "workflow": {"raw": {"too_large": True}},
                        "artifacts": [],
                    }
                ),
            ),
        ),
        evaluator_delegate=evaluator,
    )

    result = pattern.run("Select the best concept.")

    assert result.success
    assert evaluator.prompts
    assert '"concept": "concept-a"' in evaluator.prompts[0]
    assert '"workflow"' not in evaluator.prompts[0]


def test_nominal_team_pattern_reports_no_candidates() -> None:
    pattern = NominalTeamPattern(
        team_members=(
            NominalTeamPattern.MemberSpec(
                member_id="designer_a",
                delegate=_StaticTeamAgent(output={}),
            ),
            NominalTeamPattern.MemberSpec(
                member_id="designer_b",
                delegate=_StaticTeamAgent(output={"model_text": ""}),
            ),
        ),
        evaluator_delegate=lambda _context: {"best_index": 0},
    )

    result = pattern.run("Select the best concept.")

    assert not result.success
    assert result.output["terminated_reason"] == "no_candidates"
    assert result.output["details"]["candidate_count"] == 0
    assert result.output["final_output"] == {}


def test_nominal_team_pattern_internal_helpers_cover_generation_and_selection_paths() -> None:
    pattern = NominalTeamPattern(
        team_members=(
            NominalTeamPattern.MemberSpec(
                member_id="member_a",
                delegate=_StaticTeamAgent(output={"concept": "a"}),
                prompt_template="Task={task}\nMember={member_id}",
            ),
            NominalTeamPattern.MemberSpec(
                member_id="member_b",
                delegate=_StaticTeamAgent(output={"concept": "b"}),
            ),
        ),
        evaluator_delegate=lambda _context: {"best_index": 0},
    )

    calls = pattern._build_generation_calls(prompt="Select a concept.")
    assert calls[0].prompt == "Task=Select a concept.\nMember=member_a"
    assert '"member_id": "member_b"' in calls[1].prompt

    nominal_team_impl._validate_member_prompt_template(
        prompt_template="Literal={} Task={task}",
        member_id="member_a",
    )

    assert nominal_team_impl._extract_generation_result({}) is None
    assert nominal_team_impl._extract_generation_result({"dependency_results": []}) is None
    assert nominal_team_impl._extract_generation_result({"dependency_results": {"nominal_team_generation": []}}) is None

    candidate_results = [
        {
            "member_id": "member_a",
            "candidate_index": 0,
            "team_index": 0,
            "candidate": {"concept": "a"},
            "output": {"concept": "a"},
        },
        {
            "member_id": "member_b",
            "candidate_index": 1,
            "team_index": 1,
            "candidate": {"concept": "b"},
            "output": {"concept": "b"},
        },
    ]

    assert nominal_team_impl._normalize_evaluator_output({"final_output": {"best_member_id": "member_a"}}) == {
        "best_member_id": "member_a"
    }
    assert nominal_team_impl._normalize_evaluator_output({"final_output": [0.1, 0.2]}) == {"scores": [0.1, 0.2]}
    assert nominal_team_impl._normalize_evaluator_output({"final_output": '{"best_index": 1}'}) == {"best_index": 1}
    assert nominal_team_impl._normalize_evaluator_output({"final_output": 1}) == {"best_index": 1}
    assert nominal_team_impl._normalize_evaluator_output({"model_text": '{"selected_member_id":"member_b"}'}) == {
        "selected_member_id": "member_b"
    }

    assert (
        nominal_team_impl._resolve_selected_result(
            evaluator_output={"selected_member_id": "member_b"},
            candidate_results=candidate_results,
        )
        == candidate_results[1]
    )
    assert (
        nominal_team_impl._resolve_selected_result(
            evaluator_output={"scores": [0.4, 0.9]},
            candidate_results=candidate_results,
        )
        == candidate_results[1]
    )
    assert (
        nominal_team_impl._resolve_selected_result(
            evaluator_output={"winner": 7},
            candidate_results=candidate_results,
        )
        is None
    )

    assert nominal_team_impl._normalize_score_map(
        evaluator_output={"scores": {"member_a": "0.7", "member_b": 0.2}},
        candidate_results=candidate_results,
    ) == {"member_a": 0.7, "member_b": 0.2}
    assert nominal_team_impl._normalize_score_map(
        evaluator_output={"scores": [0.3, "bad", 0.9]},
        candidate_results=candidate_results,
    ) == {"member_a": 0.3}
    assert (
        nominal_team_impl._normalize_score_map(
            evaluator_output={"scores": "bad"},
            candidate_results=candidate_results,
        )
        == {}
    )


def test_nominal_team_pattern_internal_helpers_cover_candidate_and_scalar_branches() -> None:
    assert nominal_team_impl._extract_candidate({"final_output": "draft"}) == {"value": "draft"}
    assert nominal_team_impl._extract_candidate({"candidate": 3}) == {"value": 3}
    assert nominal_team_impl._extract_candidate({"model_text": '{"concept":"json"}'}) == {"concept": "json"}
    assert nominal_team_impl._extract_candidate({"model_text": "[1, 2]"}) == {"text": "[1, 2]"}
    assert nominal_team_impl._extract_candidate({"notes": ["a"], "flag": False}) == {
        "notes": ["a"],
        "flag": False,
    }

    assert nominal_team_impl._parse_json_context("not-json") == {"task": "not-json"}
    assert nominal_team_impl._parse_json_context('{"task":"x","depth":1}') == {"task": "x", "depth": 1}
    assert nominal_team_impl._parse_json_context("[1, 2]") == {"task": "[1, 2]"}

    assert nominal_team_impl._parse_evaluator_text("winner_a") == {"best_member_id": "winner_a"}
    assert nominal_team_impl._parse_evaluator_text("") == {}
    assert nominal_team_impl._parse_candidate_from_text("freeform idea") == {"text": "freeform idea"}
    assert nominal_team_impl._parse_candidate_from_text("") == {}

    assert nominal_team_impl._normalize_candidate_value({"a": 1}) == {"a": 1}
    assert nominal_team_impl._normalize_candidate_value(2) == {"value": 2}
    assert nominal_team_impl._normalize_candidate_value(object()) == {}

    assert nominal_team_impl._has_meaningful_output({"items": [1]}) is True
    assert nominal_team_impl._has_meaningful_output({"flag": True}) is True
    assert nominal_team_impl._has_meaningful_output({"empty": ""}) is False

    result_with_output_error = ExecutionResult(
        output={"error": "from-output"},
        success=False,
        tool_results=[],
        model_response=None,
    )
    assert (
        nominal_team_impl._extract_execution_error(
            result=result_with_output_error,
            fallback="fallback",
        )
        == "from-output"
    )
    assert (
        nominal_team_impl._extract_execution_error(
            result=ExecutionResult(output={}, success=False, tool_results=[], model_response=None),
            fallback="fallback",
        )
        == "fallback"
    )

    assert nominal_team_impl._coerce_optional_int(True) == 1
    assert nominal_team_impl._coerce_optional_int(2.9) == 2
    assert nominal_team_impl._coerce_optional_int("bad") is None
    assert nominal_team_impl._coerce_optional_float(True) == 1.0
    assert nominal_team_impl._coerce_optional_float("4.5") == 4.5
    assert nominal_team_impl._coerce_optional_float("bad") is None
    assert nominal_team_impl._safe_int(None) == 0
    assert nominal_team_impl._as_dict([]) == {}
    assert nominal_team_impl._as_list((1, 2)) == [1, 2]
    assert nominal_team_impl._as_list({}) == []
    assert nominal_team_impl._value_or_none(None) is None
    assert nominal_team_impl._value_or_none(3) == 3

    json_ready = nominal_team_impl._json_ready({"values": (1, object())})
    assert json_ready["values"][0] == 1
    assert isinstance(json_ready["values"][1], str)


def test_nominal_team_pattern_internal_adapter_and_result_builder_paths() -> None:
    adapter = nominal_team_impl._NominalTeamEvaluatorCallableAdapter(lambda _context: {"best_index": 0})
    mapping_result = adapter.run(context={"prompt": '{"task":"x"}'})
    assert mapping_result.success
    assert mapping_result.output == {"best_index": 0}

    string_result = nominal_team_impl._NominalTeamEvaluatorCallableAdapter(lambda _context: "member_b").run(
        context={"prompt": "{}"}
    )
    assert string_result.output == {"best_member_id": "member_b"}

    int_result = nominal_team_impl._NominalTeamEvaluatorCallableAdapter(lambda _context: 1).run(
        context={"prompt": "{}"}
    )
    assert int_result.output == {"best_index": 1}

    sequence_result = nominal_team_impl._NominalTeamEvaluatorCallableAdapter(lambda _context: [0.2, "bad", 0.9]).run(
        context={"prompt": "{}"}
    )
    assert sequence_result.output == {"scores": [0.2, 0.9]}

    empty_result = nominal_team_impl._NominalTeamEvaluatorCallableAdapter(lambda _context: object()).run(
        context={"prompt": "{}"}
    )
    assert empty_result.output == {}

    failing_result = nominal_team_impl._NominalTeamEvaluatorCallableAdapter(
        lambda _context: (_ for _ in ()).throw(RuntimeError("boom"))
    ).run(context={"prompt": "{}"})
    assert not failing_result.success
    assert failing_result.output["error"] == "boom"

    agent = _StaticEvaluatorAgent(output={"best_index": 0})
    assert nominal_team_impl._normalize_evaluator_delegate(agent) is agent
    assert isinstance(
        nominal_team_impl._normalize_evaluator_delegate(lambda _context: {"best_index": 0}),
        nominal_team_impl._NominalTeamEvaluatorCallableAdapter,
    )
    assert nominal_team_impl._is_delegate_like(agent) is True
    assert nominal_team_impl._is_delegate_like(object()) is False

    missing_selection_result = nominal_team_impl._build_nominal_team_result(
        workflow_result=ExecutionResult(
            output={"artifacts": []},
            success=False,
            tool_results=[],
            model_response=None,
            step_results={},
        ),
        request_id="req-missing",
        dependencies={},
        team_members=(
            NominalTeamPattern.MemberSpec(
                member_id="member_a",
                delegate=_StaticTeamAgent(output={"concept": "a"}),
            ),
        ),
    )
    assert not missing_selection_result.success
    assert missing_selection_result.output["terminated_reason"] == "generation_failure"
