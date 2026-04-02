"""Tests for the simulated annealing pattern scaffold."""

from __future__ import annotations

import math

import pytest

from design_research_agents._implementations import SimulatedAnnealingPattern
from design_research_agents._implementations._patterns._simulated_annealing_pattern import (
    ExponentialSchedule,
    LinearSchedule,
    LogarithmicSchedule,
)
from design_research_agents._runtime._patterns import MODE_SIMULATED_ANNEALING


def test_temperature_schedules_apply_expected_decay() -> None:
    assert LinearSchedule(alpha=2.5).get_temperature(10.0, 3) == 2.5
    assert math.isclose(ExponentialSchedule(alpha=0.5).get_temperature(8.0, 2), 2.0)
    assert math.isclose(LogarithmicSchedule(c=10.0, d=2.0).get_temperature(100.0, 1), 10.0 / math.log(3.0))


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
