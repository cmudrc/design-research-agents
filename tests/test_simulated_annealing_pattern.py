"""Tests for the simulated annealing pattern."""

from __future__ import annotations

import math
import statistics

import pytest

from design_research_agents._implementations import SimulatedAnnealingPattern
from design_research_agents._implementations._patterns._simulated_annealing_pattern import (
    AdaptiveSchedule,
    ExponentialSchedule,
    LinearSchedule,
    LogarithmicSchedule,
)

# ---------------------- Temperature schedule tests ----------------------


def test_temperature_schedules_apply_expected_decay() -> None:
    assert LinearSchedule(alpha=2.5).get_temperature(10.0, 3) == 2.5
    assert math.isclose(ExponentialSchedule(alpha=0.5).get_temperature(8.0, 2), 2.0)
    assert math.isclose(LogarithmicSchedule(c=10.0, d=2.0).get_temperature(100.0, 1), 10.0 / math.log(3.0))
    assert math.isclose(
        AdaptiveSchedule(delta=0.5).get_temperature(
            100.0, 0, current_temperature=5.0, objective_value_history=[0.0, 10.0]
        ),
        5.0 * (1.0 - 5.0 * 0.5 / statistics.variance([0.0, 10.0])),
    )


def test_linear_schedule_validates_alpha_non_negative() -> None:
    with pytest.raises(ValueError, match="alpha"):
        LinearSchedule(alpha=-1.0)


def test_logarithmic_schedule_validates_d_greater_than_one() -> None:
    with pytest.raises(ValueError, match="d"):
        LogarithmicSchedule(c=1.0, d=1.0)
    
    with pytest.raises(ValueError, match="d"):
        LogarithmicSchedule(c=1.0, d=0.5)


def test_exponential_schedule_validates_alpha_range() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ExponentialSchedule(alpha=0.0)

    with pytest.raises(ValueError, match="alpha"):
        ExponentialSchedule(alpha=1.0)


def test_adaptive_schedule_derives_delta_from_objective_value_history() -> None:
    objective_value_history = [0.0, 20.0, 10.0]
    t_k = 5.0
    sched = AdaptiveSchedule(mu=5.0)
    result = sched.get_temperature(100.0, 0, current_temperature=t_k, objective_value_history=objective_value_history)
    expected_delta = statistics.stdev(objective_value_history) / 5.0
    sigma_sq = statistics.variance(objective_value_history)
    expected = t_k * (1 - t_k * expected_delta / sigma_sq)
    assert math.isclose(result, expected)


def test_adaptive_schedule_falls_back_when_history_too_short() -> None:
    sched = AdaptiveSchedule(delta=1.5)
    assert sched.get_temperature(100.0, 0, current_temperature=80.0, objective_value_history=[]) == 80.0
    assert sched.get_temperature(100.0, 0, current_temperature=80.0, objective_value_history=[90.0]) == 80.0
    assert sched.get_temperature(100.0, 0) == 100.0


def test_adaptive_schedule_falls_back_when_variance_is_zero() -> None:
    sched = AdaptiveSchedule(delta=1.5)
    assert sched.get_temperature(100.0, 0, current_temperature=80.0, objective_value_history=[50.0, 50.0, 50.0]) == 80.0


def test_adaptive_schedule_falls_back_when_factor_exceeds_one() -> None:
    sched = AdaptiveSchedule(delta=100.0)  # Large delta to force factor > 1
    assert sched.get_temperature(100.0, 0, current_temperature=50.0, objective_value_history=[0.0, 1.0]) == 50.0


def test_adaptive_schedule_delta_is_fixed_after_first_derivation() -> None:
    sched = AdaptiveSchedule(mu=5.0)
    first_history = [0.0, 10.0]
    sched.get_temperature(100.0, 0, current_temperature=5.0, objective_value_history=objective_value_history)
    first_delta = sched.delta

    second_history = [0.0, 10.0, 100.0]
    sched.get_temperature(100.0, 1, current_temperature=4.0, objective_value_history=second_history)
    assert sched.delta == first_delta


# ---------------------- Input validation tests ----------------------


def test_pattern_validates_max_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            initial_state={"x": 0.0},
            max_iterations=0,
        )


def test_pattern_validates_initial_temperature() -> None:
    with pytest.raises(ValueError, match="initial_temperature"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            initial_state={"x": 0.0},
            initial_temperature=-10.0,
        )


def test_pattern_validates_convergence_threshold() -> None:
    with pytest.raises(ValueError, match="convergence_threshold"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            initial_state={"x": 0.0},
            convergence_threshold=-1e-6,
        )


def test_pattern_validates_convergence_steps() -> None:
    with pytest.raises(ValueError, match="convergence_steps"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            initial_state={"x": 0.0},
            convergence_steps=0,
        )


def test_pattern_rejects_initial_state_violating_constraints() -> None:
    with pytest.raises(ValueError, match="initial_state"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            constraints=[lambda state: state["x"] > 0],
            initial_state={"x": -1.0},
        )


def test_pattern_rejects_non_string_keys_in_initial_state() -> None:
    with pytest.raises(ValueError, match="initial_state"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            initial_state={0: 1.0},
        )


def test_pattern_requires_mutually_exclusive_neighbor_or_modifications_delegate() -> None:
    with pytest.raises(ValueError, match="neighbor_delegate"):
        SimulatedAnnealingPattern(
            objective_delegate=lambda state: 0.0,
            initial_state={"x": 0.0},
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            modifications_delegate=lambda state: (state, {"x": 1.0}),
            objective_delegate=lambda state: 0.0,
            initial_state={"x": 0.0},
        )


def test_pattern_requires_exactly_one_initial_state_option() -> None:
    with pytest.raises(ValueError, match="initial_state"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            initial_state={"x": 0.0},
            initial_state_generator=lambda: {"x": 1.0},
        )


def test_pattern_rejects_initial_state_missing_expected_keys() -> None:
    with pytest.raises(ValueError, match="missing expected keys"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            initial_state={"y": 1.0},
            expected_keys={"x", "y"},
        )


def test_pattern_rejects_initial_state_failing_state_validator() -> None:
    with pytest.raises(ValueError, match="state_validator"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: state,
            objective_delegate=lambda state: 0.0,
            initial_state={"x": -1.0},
            state_validator=lambda state: state["x"] >= 0,
        )


# ---------------------- Delegate behavior tests ----------------------


def test_pattern_accepts_initial_state_generator() -> None:
    def generator() -> dict[str, object]:
        return {"x": 1.0}

    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda state: state,
        objective_delegate=lambda state: 0.0,
        initial_state_generator=generator,
    )
    assert pattern._initial_state is None
    assert pattern._initial_state_generator is generator


def test_pattern_validates_generated_initial_state_before_objective() -> None:
    objective_calls = 0

    def objective_delegate(state: object) -> float:
        nonlocal objective_calls
        objective_calls += 1
        return 0.0

    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda state: state,
        objective_delegate=objective_delegate,
        initial_state_generator=lambda: {"x": -1.0},
        state_validator=lambda state: state["x"] >= 0,
    )

    with pytest.raises(ValueError, match="state_validator"):
        pattern.run("test")

    assert objective_calls == 0


def test_modifications_delegate_stored_when_provided() -> None:
    def modifications_delegate(state: object) -> list[dict]:
        return [{"x": 1.0}]

    pattern = SimulatedAnnealingPattern(
        modifications_delegate=modifications_delegate,
        objective_delegate=lambda state: state["x"],
        initial_state={"x": 0.0},
    )
    assert pattern._modifications_delegate is modifications_delegate
    assert pattern._neighbor_delegate is None


# ---------------------- Objective mode tests ----------------------


def test_objective_mode_to_internal_score_minimize() -> None:
    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda state: state,
        objective_delegate=lambda state: state["x"],
        initial_state={"x": 0.0},
    )
    assert pattern._objective_mode == "minimize"
    assert pattern._to_internal_score(5.0) == 5.0
    assert pattern._to_internal_score(-3.0) == -3.0


def test_objective_mode_to_internal_score_maximize() -> None:
    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda state: state,
        objective_delegate=lambda state: state["x"],
        initial_state={"x": 0.0},
        objective_mode="maximize",
    )
    assert pattern._objective_mode == "maximize"
    assert pattern._to_internal_score(3.0) == -3.0
    assert pattern._to_internal_score(-2.0) == 2.0


# ---------------------- Execution tests ----------------------


def test_execution_with_initial_state_generator() -> None:
    def generator() -> dict[str, object]:
        return {"x": 5.0}

    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda state: state,
        objective_delegate=lambda state: state["x"],
        initial_state_generator=generator,
        max_iterations=1,
        random_seed=0,
    )
    result = pattern.run("test")
    assert result.success
    assert result.output["final_output"]["best_state"] is not None
    assert result.output["details"]["initial_state"] == {"x": 5.0}


def test_execution_validates_neighbor_state_before_objective() -> None:
    objective_inputs: list[object] = []

    def objective_delegate(state: object) -> float:
        objective_inputs.append(state["x"])
        return float(state["x"])

    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda state: {"x": -1.0},
        objective_delegate=objective_delegate,
        initial_state={"x": 0.0},
        state_validator=lambda state: state["x"] >= 0,
        max_iterations=1,
        random_seed=0,
    )

    result = pattern.run("test")

    assert not result.success
    assert objective_inputs == [0.0]
    assert "state_validator" in str(result.to_dict())


def test_execution_with_modifications_delegate() -> None:
    def delegate(state: object) -> list[dict]:
        return [{"x": state["x"] - 1.0}]

    pattern = SimulatedAnnealingPattern(
        modifications_delegate=delegate,
        objective_delegate=lambda state: state["x"],
        initial_state={"x": 10.0},
        max_iterations=3,
        random_seed=0,
    )
    result = pattern.run("test")
    assert result.success
    assert result.output["final_output"]["best_state"] is not None
    assert result.output["final_output"]["best_state"]["x"] < 10.0


def test_execution_rejects_empty_modifications_delegate_result() -> None:
    pattern = SimulatedAnnealingPattern(
        modifications_delegate=lambda state: [],
        objective_delegate=lambda state: state["x"],
        initial_state={"x": 10.0},
        max_iterations=1,
        random_seed=0,
    )

    result = pattern.run("test")

    assert not result.success
    result_payload = str(result.to_dict())
    assert "modifications_delegate must return at least one modification" in result_payload
    assert "IndexError" not in result_payload


def test_execution_with_objective_mode_maximize() -> None:
    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda state: {"x": state["x"] + 1.0},
        objective_delegate=lambda state: state["x"],
        initial_state={"x": 0.0},
        objective_mode="maximize",
        max_iterations=5,
        random_seed=0,
    )
    result = pattern.run("test")
    assert result.success
    assert result.output["final_output"]["best_state"] is not None
    assert result.output["final_output"]["best_state"]["x"] >= 0.0


def test_execution_detects_convergence_after_objective_changes_then_stabilizes() -> None:
    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda state: {"x": 1.0},
        objective_delegate=lambda state: -float(state["x"]),
        initial_state={"x": 0.0},
        initial_temperature=1.0,
        max_iterations=5,
        convergence_steps=1,
        random_seed=0,
    )

    result = pattern.run("test")

    assert result.success
    assert result.output["terminated_reason"] == "converged"
    assert result.output["final_output"]["iterations"] == 2


# Optimization test ideas to begin with:
#     Objective: maximize a polynomial function: f(x) = x^4-4x^3-2x^2+12x+1
#          Initial state: x = 2
#          Neighbor delegate: propose a new x by adding a random value from [-1, 1] to current x
#          Objective delegate: f(x) for the proposed x
#          Constraints: x must be between [-2, 2]
"""
        def polynomial(x: float) -> float:
            return x**4 - 4*x**3 - 2*x**2 + 12*x + 1

        pattern_1 = SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: {
                "x": state["x"] + random.uniform(-1, 1)
            },
            objective_delegate=lambda state: -polynomial(state["x"]),  # negate to maximize
            constraints=[
                lambda state: -2 <= state["x"] <= 2,
            ],
            initial_state={"x": 2.0},
            initial_temperature=100.0,
            max_iterations=1000,
            random_seed=42,
        )
"""

#     Objective: minimize the volume of a beam with length L, width w, and height h
#          Initial state: L=5m, w=2m, h=1m
#          Neighbor delegate: propose new dimensions by adding a random value from [-0.5, 0.5] to each dimension
#          Objective delegate: volume V = L * w * h
#          Constraints: dimensions must be positive, max stress in beam must be below 250 MPa under a load of 10000 N
"""
        P = 10000   # applied load in Newtons
        MAX_STRESS = 250e6  # 250 MPa in Pascals

        def max_bending_stress(state: dict) -> float:
            L, w, h = state["L"], state["w"], state["h"]
            return (6 * P * L) / (w * h**2)

        def max_shear_stress(state: dict) -> float:
            L, w, h = state["L"], state["w"], state["h"]
            return (3 * P) / (2 * w * h)

        pattern_2 = SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: {
                "L": state["L"] + random.uniform(-0.5, 0.5),
                "w": state["w"] + random.uniform(-0.5, 0.5),
                "h": state["h"] + random.uniform(-0.5, 0.5),
            },
            objective_delegate=lambda state: state["L"] * state["w"] * state["h"],
            constraints=[
                lambda state: state["L"] > 0,
                lambda state: state["w"] > 0,
                lambda state: state["h"] > 0,
                lambda state: max_bending_stress(state) < MAX_STRESS,
                lambda state: max_shear_stress(state) < MAX_STRESS,
            ],
            initial_state={"L": 5.0, "w": 2.0, "h": 1.0},
            initial_temperature=100.0,
            max_iterations=1000,
            random_seed=42,
        )
"""
