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
from design_research_agents._runtime._patterns import MODE_SIMULATED_ANNEALING


# ---------------------- Temperature schedule tests ----------------------

def test_temperature_schedules_apply_expected_decay() -> None:
    assert LinearSchedule(alpha=2.5).get_temperature(10.0, 3) == 2.5
    assert math.isclose(ExponentialSchedule(alpha=0.5).get_temperature(8.0, 2), 2.0)
    assert math.isclose(LogarithmicSchedule(c=10.0, d=2.0).get_temperature(100.0, 1), 10.0 / math.log(3.0))
    assert math.isclose(
        AdaptiveSchedule(delta=0.5).get_temperature(100.0, 0, 5.0, [0.0, 10.0]),
        5.0 * (1.0 - 5.0 * 0.5 / statistics.variance([0.0, 10.0]))
    )


def test_exponential_schedule_validates_alpha_range() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ExponentialSchedule(alpha=0.0)

    with pytest.raises(ValueError, match="alpha"):
        ExponentialSchedule(alpha=1.0)


def test_simulated_annealing_pattern_validates_basic_configuration() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda context: context,
            energy_delegate=lambda context: 0.0,
            initial_state={},
            max_iterations=0,
        )

    with pytest.raises(ValueError, match="initial_temperature"):
        SimulatedAnnealingPattern(
            neighbor_delegate=lambda context: context,
            energy_delegate=lambda context: 0.0,
            initial_state={},
            initial_temperature=0,
        )


def test_simulated_annealing_pattern_run_returns_structured_scaffold_result() -> None:
    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=lambda context: context,
        energy_delegate=lambda context: 0.0,
        initial_state={"design": "baseline"},
    )

    result = pattern.run("Reduce drag.")

    assert not result.success
    assert result.output["terminated_reason"] == "not_implemented"
    assert result.output["error"] == "Simulated annealing pattern is scaffolded but not implemented yet."
    assert result.output["details"]["initial_state"] == {"design": "baseline"}
    assert result.output["details"]["temperature_schedule"] == "ExponentialSchedule"
    assert result.metadata["mode"] == MODE_SIMULATED_ANNEALING


def test_adaptive_schedule_derives_delta_from_energy_history() -> None:
    energy_history = [0.0, 20.0, 10.0]
    t_k = 5.0
    sched = AdaptiveSchedule(mu=5.0)
    result = sched.get_temperature(100.0, 0, current_temperature=t_k, energy_history=energy_history)
    expected_delta = statistics.stdev(energy_history) / 5.0
    sigma_sq = statistics.variance(energy_history)
    expected = t_k * (1 - t_k * expected_delta / sigma_sq)
    assert math.isclose(result, expected)
    assert math.isclose(sched.delta, expected_delta)


def test_adaptive_schedule_falls_back_when_history_too_short() -> None:
    sched = AdaptiveSchedule(delta=1.5)
    assert sched.get_temperature(100.0, 0, current_temperature=80.0, energy_history=[]) == 80.0
    assert sched.get_temperature(100.0, 0, current_temperature=80.0, energy_history=[90.0]) == 80.0
    assert sched.get_temperature(100.0, 0) == 100.0


def test_adaptive_schedule_falls_back_when_variance_is_zero() -> None:
    sched = AdaptiveSchedule(delta=1.5)
    assert sched.get_temperature(100.0, 0, current_temperature=80.0, energy_history=[50.0, 50.0, 50.0]) == 80.0


def test_adaptive_schedule_falls_back_when_factor_exceeds_one() -> None:
    sched = AdaptiveSchedule(delta=100.0) # Large delta to force factor > 1
    assert sched.get_temperature(100.0, 0, current_temperature=50.0, energy_history=[0.0, 1.0]) == 50.0


# Optimization test ideas to begin with:
#     Objective: maximize a polynomial function: f(x) = x^4-4x^3-2x^2+12x+1
#          Initial state: x = 2
#          Neighbor delegate: propose a new x by adding a random value from [-1, 1] to current x
#          Energy delegate: compute f(x) for the proposed x
#          Constraints: x must be between [-2, 2]
"""
        def polynomial(x: float) -> float:
            return x**4 - 4*x**3 - 2*x**2 + 12*x + 1

        pattern_1 = SimulatedAnnealingPattern(
            neighbor_delegate=lambda state: {
                "x": state["x"] + random.uniform(-1, 1)
            },
            energy_delegate=lambda state: -polynomial(state["x"]),  # negate to maximize
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
#          Energy delegate: compute the volume V = L * w * h
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
            energy_delegate=lambda state: state["L"] * state["w"] * state["h"],
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
