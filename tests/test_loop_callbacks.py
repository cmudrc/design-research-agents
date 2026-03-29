"""Tests for shared loop callback helpers."""

from __future__ import annotations

import pytest

from design_research_agents._contracts._workflow import WorkflowStepResult
from design_research_agents._runtime._patterns import build_loop_callbacks, wrap_iteration_handler
from design_research_agents.workflow import ExecutionResult


def test_wrap_iteration_handler_requires_mapping_output() -> None:
    wrapped = wrap_iteration_handler(lambda _context: {"ok": True}, error_prefix="loop")
    assert wrapped({}) == {"ok": True}

    bad = wrap_iteration_handler(lambda _context: "bad", error_prefix="loop")
    with pytest.raises(ValueError, match="must return a mapping"):
        bad({})


def test_build_loop_callbacks_default_predicate_and_reducer() -> None:
    wrapped = wrap_iteration_handler(
        lambda _context: {"value": 2, "should_continue": False},
        error_prefix="iter",
    )
    callbacks = build_loop_callbacks(
        iteration_step_id="iter",
        iteration_handler=wrapped,
    )

    assert callbacks.continue_predicate(1, {"should_continue": True})
    assert not callbacks.continue_predicate(2, {"should_continue": False})

    iteration_result = ExecutionResult(
        output={},
        success=True,
        step_results={
            "iter": WorkflowStepResult(
                step_id="iter",
                status="completed",
                success=True,
                output={"value": 7, "should_continue": False},
            )
        },
    )
    reduced = callbacks.state_reducer({"value": 1}, iteration_result, 1)
    assert reduced == {"value": 7, "should_continue": False}
