from __future__ import annotations

import inspect
from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from design_research_agents import SeededRandomBaselineAgent
from design_research_agents._contracts._termination import (
    TERMINATED_COMPLETED,
    TERMINATED_MAX_STEPS_REACHED,
    TERMINATED_STEP_FAILURE,
)
from design_research_agents._implementations._agents import (
    _seeded_random_baseline_agent as seeded_random_impl,
)


@dataclass(frozen=True)
class _Metadata:
    problem_id: str


@dataclass(frozen=True)
class _DecisionCandidate:
    values: dict[str, float]


@dataclass(frozen=True)
class _GrammarTransition:
    rule_name: str
    parameters: tuple[tuple[str, object], ...]
    next_state: dict[str, object]


@dataclass(frozen=True)
class _OptimizationEvaluation:
    objective_value: float
    total_constraint_violation: float
    max_constraint_violation: float
    is_feasible: bool
    higher_is_better: bool


@dataclass(frozen=True)
class _RunSpec:
    seed: int
    run_id: str = "run-001"


@dataclass(frozen=True)
class _Condition:
    condition_id: str


class _DecisionProblem:
    def __init__(self) -> None:
        self.metadata = _Metadata(problem_id="stub-decision")
        self._candidates = (
            _DecisionCandidate({"fan_count": 4.0, "fin_gap_mm": 2.5}),
            _DecisionCandidate({"fan_count": 6.0, "fin_gap_mm": 2.0}),
            _DecisionCandidate({"fan_count": 8.0, "fin_gap_mm": 1.5}),
        )

    def iter_candidates(self) -> tuple[_DecisionCandidate, ...]:
        return self._candidates


class _GrammarProblem:
    def __init__(self) -> None:
        self.metadata = _Metadata(problem_id="stub-grammar")

    def initial_state(self) -> dict[str, object]:
        return {"depth": 0, "history": []}

    def enumerate_transitions(
        self,
        state: dict[str, object],
    ) -> tuple[_GrammarTransition, ...]:
        depth = int(state["depth"])
        history = list(state["history"])
        if depth >= 3:
            return ()
        return (
            _GrammarTransition(
                rule_name="add_left",
                parameters=(("label", "L"),),
                next_state={"depth": depth + 1, "history": [*history, "L"]},
            ),
            _GrammarTransition(
                rule_name="add_right",
                parameters=(("label", "R"),),
                next_state={"depth": depth + 1, "history": [*history, "R"]},
            ),
            _GrammarTransition(
                rule_name="add_center",
                parameters=(("label", "C"),),
                next_state={"depth": depth + 1, "history": [*history, "C"]},
            ),
        )


class _OptimizationProblem:
    def __init__(self) -> None:
        self.metadata = _Metadata(problem_id="stub-optimization")

    def generate_initial_solution(self, seed: int | None = None) -> list[float]:
        resolved_seed = 0 if seed is None else int(seed)
        return [
            round((resolved_seed % 5) / 10.0, 3),
            round(((resolved_seed + 3) % 7) / 10.0, 3),
        ]

    def evaluate(self, candidate: list[float]) -> _OptimizationEvaluation:
        objective_value = round(sum(candidate), 4)
        return _OptimizationEvaluation(
            objective_value=objective_value,
            total_constraint_violation=0.0,
            max_constraint_violation=0.0,
            is_feasible=True,
            higher_is_better=False,
        )


class _NoTransitionGrammarProblem(_GrammarProblem):
    def initial_state(self) -> dict[str, object]:
        return {"depth": 3, "history": ["done"]}


class _OptimizationProblemWithFailingPreview(_OptimizationProblem):
    def evaluate(self, candidate: list[float]) -> _OptimizationEvaluation:
        del candidate
        raise RuntimeError("preview failed")


class _PositionalSeedOptimizationProblem:
    def generate_initial_solution(self, seed: int) -> list[int]:
        return [seed]


class _NoArgOptimizationProblem:
    def generate_initial_solution(self) -> list[int]:
        return [42]


def test_seeded_random_baseline_matches_standard_delegate_signature() -> None:
    run_parameters = inspect.signature(SeededRandomBaselineAgent.run).parameters
    compile_parameters = inspect.signature(SeededRandomBaselineAgent.compile).parameters

    assert tuple(run_parameters) == ("self", "prompt", "request_id", "dependencies")
    assert tuple(compile_parameters) == ("self", "prompt", "request_id", "dependencies")
    assert run_parameters["prompt"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert compile_parameters["prompt"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for parameter_name in ("request_id", "dependencies"):
        assert run_parameters[parameter_name].kind is inspect.Parameter.KEYWORD_ONLY
        assert compile_parameters[parameter_name].kind is inspect.Parameter.KEYWORD_ONLY


def test_seeded_random_baseline_compile_exposes_standard_delegate_workflow() -> None:
    agent = SeededRandomBaselineAgent(seed=5)
    compiled = agent.compile(
        "Sample one control condition.",
        dependencies={"problem": _DecisionProblem()},
    )

    mermaid = compiled.to_mermaid(direction="LR")

    assert "seeded_random_baseline" in mermaid
    assert agent.compile_to_mermaid(direction="LR") == mermaid


def test_decision_baseline_is_deterministic_for_fixed_seed() -> None:
    agent = SeededRandomBaselineAgent()
    problem = _DecisionProblem()

    first = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem, "seed": 5},
    )
    second = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem, "seed": 5},
    )

    assert first.output == second.output
    assert first.metadata["selected_index"] == second.metadata["selected_index"]


def test_decision_baseline_changes_when_seed_changes() -> None:
    agent = SeededRandomBaselineAgent()
    problem = _DecisionProblem()

    first = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem, "seed": 1},
    )
    second = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem, "seed": 5},
    )

    assert first.output["final_output"] != second.output["final_output"]


def test_decision_output_is_drawn_from_candidate_set() -> None:
    agent = SeededRandomBaselineAgent(seed=5)
    problem = _DecisionProblem()

    result = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem},
    )
    admissible_outputs = [dict(candidate.values) for candidate in problem.iter_candidates()]

    assert result.output["final_output"] in admissible_outputs
    assert result.output["terminated_reason"] == TERMINATED_COMPLETED


def test_seed_precedence_prefers_dependency_seed_then_run_spec_then_agent_default() -> None:
    problem = _DecisionProblem()
    agent = SeededRandomBaselineAgent(seed=7)
    run_spec = _RunSpec(seed=5)

    explicit = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem, "seed": 1, "run_spec": run_spec},
    )
    from_run_spec = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem, "run_spec": run_spec},
    )
    from_agent_default = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem},
    )

    assert explicit.metadata["seed"] == 1
    assert explicit.metadata["seed_source"] == "dependencies.seed"
    assert from_run_spec.metadata["seed"] == 5
    assert from_run_spec.metadata["seed_source"] == "dependencies.run_spec.seed"
    assert from_agent_default.metadata["seed"] == 7
    assert from_agent_default.metadata["seed_source"] == "agent_default"


def test_problem_packet_wrapper_is_supported() -> None:
    problem = _DecisionProblem()
    packet = {
        "problem_id": "wrapped-decision",
        "payload": {"problem_object": problem},
    }
    agent = SeededRandomBaselineAgent(seed=5)

    result = agent.run(
        "Sample a control candidate.",
        dependencies={"problem_packet": packet},
    )

    assert result.success is True
    assert result.metadata["problem_id"] == "wrapped-decision"
    assert result.metadata["family"] == "decision"


def test_grammar_rollout_respects_max_steps_and_records_transitions() -> None:
    agent = SeededRandomBaselineAgent(seed=5, grammar_max_steps=2)
    problem = _GrammarProblem()
    condition = _Condition(condition_id="cond-1")

    result = agent.run(
        "Sample a grammar rollout.",
        dependencies={"problem": problem, "condition": condition},
    )

    assert result.success is True
    assert result.metadata["family"] == "grammar"
    assert result.metadata["steps_requested"] == 2
    assert result.metadata["steps_executed"] == 2
    assert len(result.metadata["transitions"]) == 2
    assert len(result.output["final_output"]["state"]["history"]) == 2
    assert result.output["metrics"]["steps_executed"] == 2
    assert result.output["terminated_reason"] == TERMINATED_MAX_STEPS_REACHED
    assert result.metadata["condition_id"] == "cond-1"


def test_grammar_rollout_reports_when_no_transitions_are_available() -> None:
    agent = SeededRandomBaselineAgent(seed=5, grammar_max_steps=2)
    problem = _NoTransitionGrammarProblem()

    result = agent.run(
        "Sample a grammar rollout.",
        dependencies={"problem": problem},
    )

    assert result.success is True
    assert result.metadata["family"] == "grammar"
    assert result.metadata["steps_executed"] == 0
    assert result.metadata["transitions"] == []
    assert result.output["terminated_reason"] == "no_transitions_available"
    assert result.output["events"][0]["event_type"] == "baseline_transition_skipped"


def test_optimization_preview_evaluation_is_preserved() -> None:
    agent = SeededRandomBaselineAgent(seed=11)
    problem = _OptimizationProblem()

    result = agent.run(
        "Sample an optimization candidate.",
        dependencies={"problem": problem},
    )

    assert result.success is True
    assert result.metadata["family"] == "optimization"
    assert result.output["final_output"]["candidate"] == [0.1, 0.0]
    assert result.output["metrics"]["objective_value"] == 0.1
    assert result.metadata["preview_evaluation"]["is_feasible"] is True


def test_optimization_preview_evaluation_errors_are_recorded_without_failing() -> None:
    agent = SeededRandomBaselineAgent(seed=11)
    problem = _OptimizationProblemWithFailingPreview()

    result = agent.run(
        "Sample an optimization candidate.",
        dependencies={"problem": problem},
    )

    assert result.success is True
    assert result.metadata["family"] == "optimization"
    assert result.output["metrics"] == {}
    assert result.metadata["preview_evaluation_error"] == "preview failed"


def test_missing_problem_dependency_returns_structured_failure_result() -> None:
    agent = SeededRandomBaselineAgent()

    result = agent.run("Sample a control candidate.")

    assert result.success is False
    assert result.error == (
        "SeededRandomBaselineAgent requires `dependencies['problem']` or "
        "`dependencies['problem_packet']` with `payload['problem_object']`."
    )
    assert result.output["terminated_reason"] == TERMINATED_STEP_FAILURE


def test_optional_real_problem_integration_path_for_decision_problem() -> None:
    derp = pytest.importorskip("design_research_problems")
    problem = derp.get_problem("decision_laptop_design_profit_maximization")
    agent = SeededRandomBaselineAgent(seed=17)

    result = agent.run(
        "Sample a control candidate.",
        dependencies={"problem": problem},
    )
    evaluation = problem.evaluate(result.output["final_output"])

    assert isinstance(result.output["final_output"], dict)
    assert evaluation.objective_value >= 0.0


def test_seeded_random_helper_branches_cover_fallbacks_and_payload_helpers() -> None:
    assert seeded_random_impl._generate_initial_solution(
        problem=_PositionalSeedOptimizationProblem(),
        seed=7,
    ) == [7]
    assert seeded_random_impl._generate_initial_solution(
        problem=_NoArgOptimizationProblem(),
        seed=7,
    ) == [42]
    assert seeded_random_impl._decision_output_payload("choice-a") == {"choice_key": "choice-a"}
    assert seeded_random_impl._decision_output_payload({"x": 1}) == {"x": 1}
    assert seeded_random_impl._coerce_seed_like(3.9) == 3
    assert seeded_random_impl._coerce_seed_like("8") == 8


def test_seeded_random_rejects_invalid_configuration_and_problem_contracts() -> None:
    with pytest.raises(ValueError, match="grammar_max_steps"):
        SeededRandomBaselineAgent(grammar_max_steps=0)
    with pytest.raises(TypeError, match="supports only decision, grammar, and optimization"):
        seeded_random_impl._detect_problem_family(object())

    empty_decision = SimpleNamespace(iter_candidates=lambda: ())
    with pytest.raises(ValueError, match="at least one candidate"):
        SeededRandomBaselineAgent()._run_decision(problem=empty_decision, seed=1, metadata={})

    invalid_transition = SimpleNamespace(next_state=None)
    grammar = SimpleNamespace(initial_state=lambda: {}, enumerate_transitions=lambda _state: (invalid_transition,))
    with pytest.raises(TypeError, match="must expose `next_state`"):
        SeededRandomBaselineAgent()._run_grammar(problem=grammar, seed=1, metadata={})


def test_seeded_random_problem_resolution_and_metadata_helpers_cover_nested_shapes() -> None:
    problem = _DecisionProblem()
    assert (
        seeded_random_impl._resolve_problem_object(problem_packet=None, problem={"problem_object": problem}) is problem
    )
    assert (
        seeded_random_impl._resolve_problem_object(
            problem_packet={"payload": {"problem_object": problem}},
            problem=None,
        )
        is problem
    )
    assert seeded_random_impl._resolve_problem_object(problem_packet=None, problem=None) is None
    assert seeded_random_impl._nested_problem_object({"payload": "invalid"}) is None

    assert seeded_random_impl._resolve_seed(explicit_seed=2, run_spec=None, default_seed=3) == 2
    assert seeded_random_impl._resolve_seed(explicit_seed=None, run_spec={"seed": True}, default_seed=3) == 1
    assert seeded_random_impl._resolve_seed(explicit_seed=None, run_spec={}, default_seed=3) == 3
    assert seeded_random_impl._resolve_seed(explicit_seed=None, run_spec={}, default_seed=None) == 0
    assert seeded_random_impl._seed_source(explicit_seed=2, run_spec=None, default_seed=3) == "dependencies.seed"
    assert seeded_random_impl._seed_source(explicit_seed=None, run_spec={"seed": 2}, default_seed=3) == (
        "dependencies.run_spec.seed"
    )
    assert seeded_random_impl._seed_source(explicit_seed=None, run_spec={}, default_seed=3) == "agent_default"
    assert seeded_random_impl._seed_source(explicit_seed=None, run_spec={}, default_seed=None) == "implicit_default"

    direct_id = SimpleNamespace(problem_id="direct")
    assert seeded_random_impl._resolve_problem_id(problem_packet=None, problem=direct_id) == "direct"
    assert seeded_random_impl._resolve_problem_id(problem_packet=None, problem=object()) == "object"


def test_seeded_random_payload_normalizers_cover_object_and_sequence_variants() -> None:
    assert seeded_random_impl._decision_output_payload(SimpleNamespace(values={"a": 1})) == {"a": 1}
    assert seeded_random_impl._decision_output_payload(object())["candidate"].startswith("<object object at")
    assert seeded_random_impl._transition_parameters_payload({"a": 1}) == {"a": 1}
    assert seeded_random_impl._transition_parameters_payload((("a", 1), "invalid", ("bad",))) == {"a": 1}
    assert seeded_random_impl._transition_parameters_payload(object()) == {}
    assert seeded_random_impl._optimization_metrics(
        {
            "objective_value": 2,
            "total_constraint_violation": "invalid",
            "is_feasible": True,
            "higher_is_better": "invalid",
        }
    ) == {"objective_value": 2.0, "is_feasible": True}
    assert seeded_random_impl._normalize_sequence([1]) == [1]
    assert seeded_random_impl._normalize_sequence((1,)) == [1]
    assert seeded_random_impl._normalize_sequence("invalid") == []

    class _ArrayLike:
        def tolist(self) -> list[int]:
            return [1, 2]

    Point = namedtuple("Point", ["x", "y"])
    assert seeded_random_impl._json_safe_value(_ArrayLike()) == [1, 2]
    assert seeded_random_impl._json_safe_value({"point": Point(1, 2)}) == {"point": [1, 2]}
    assert seeded_random_impl._json_safe_value(_DecisionCandidate({"a": 1})) == {"values": {"a": 1}}
    assert seeded_random_impl._json_safe_value(SimpleNamespace(public=1, _private=2)) == {"public": 1}
    assert seeded_random_impl._value_from_object(None, "value") is None
    assert seeded_random_impl._value_from_object({"value": 1}, "value") == 1
    assert seeded_random_impl._coerce_seed_like(True) == 1
    assert seeded_random_impl._coerce_seed_like(2) == 2
    assert seeded_random_impl._coerce_seed_like(None) is None
