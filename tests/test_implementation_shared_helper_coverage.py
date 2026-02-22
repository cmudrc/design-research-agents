from __future__ import annotations

from types import SimpleNamespace

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import ToolResult, ToolSpec
from design_research_agents._contracts._workflow import WorkflowStepResult
from design_research_agents._implementations._shared._agent_internal import (
    _multi_step_json_runtime_helpers as json_runtime,
)
from design_research_agents._implementations._shared._agent_internal import (
    _workflow_loop_orchestration as loop_orchestration,
)
from design_research_agents._implementations._shared._workflow_internal import (
    _planner_executor_helpers as planner_helpers,
)
from design_research_agents._runtime._common import _run_defaults as canonical_run_defaults
from design_research_agents._runtime._common import _run_defaults as run_defaults
from design_research_agents._runtime._patterns import _loop_state as loop_state
from design_research_agents._runtime._patterns._run_context import (
    WorkflowBudgetTracker,
)


def _tool_spec(name: str = "calculator") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


def _planner_callbacks(plan_steps: list[object]) -> planner_helpers.PlanExecuteLoopCallbacks:
    return planner_helpers.PlanExecuteLoopCallbacks(
        prompt="task",
        plan_steps=plan_steps,  # type: ignore[arg-type]
        executor_step_prompt_template=("$task_prompt|$step_id|$instruction|$success_criteria|$prior_step_outputs_json"),
        request_id="req",
        dependencies={},
        budget_tracker=WorkflowBudgetTracker(),
        runtime_tool_specs={"calculator": _tool_spec()},
        initial_model_response=None,
    )


def test_run_defaults_and_loop_state_helpers_cover_error_and_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert run_defaults.merge_dependencies(
        default_dependencies={"a": 1},
        run_dependencies=None,
    ) == {"a": 1}
    assert run_defaults.merge_dependencies(
        default_dependencies={"a": 1},
        run_dependencies={"b": 2},
    ) == {"a": 1, "b": 2}

    assert run_defaults.normalize_request_id_prefix(None) is None
    assert run_defaults.normalize_request_id_prefix(" req ") == "req"
    with pytest.raises(ValueError, match="non-empty"):
        run_defaults.normalize_request_id_prefix("   ")

    assert (
        run_defaults.resolve_request_id_with_prefix(
            request_id="provided",
            default_prefix="ignored",
        )
        == "provided"
    )
    assert run_defaults.resolve_request_id_with_prefix(request_id=" ", default_prefix=None) == " "
    monkeypatch.setattr(canonical_run_defaults, "uuid4", lambda: SimpleNamespace(hex="abc123"))
    assert (
        run_defaults.resolve_request_id_with_prefix(
            request_id=" ",
            default_prefix="req",
        )
        == "req:abc123"
    )

    assert loop_state.normalize_mapping(1) == {}
    assert loop_state.normalize_mapping({"k": 1}) == {"k": 1}
    assert loop_state.normalize_mapping_records("x") == []
    assert loop_state.normalize_mapping_records([{"a": 1}, "bad", {"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert loop_state.parse_loop_iteration(3, error_prefix="iter") == 3
    assert loop_state.parse_loop_iteration("4", error_prefix="iter") == 4
    with pytest.raises(ValueError, match="must be an integer"):
        loop_state.parse_loop_iteration("", error_prefix="iter")


def test_workflow_loop_orchestration_covers_iteration_parsing_and_failure_paths() -> None:
    successful = loop_orchestration.run_workflow_loop(
        max_iterations=4,
        initial_state={"count": 0},
        continue_predicate=lambda iteration, state: iteration <= 2,
        iteration_handler=lambda iteration, state: {"count": int(state.get("count", 0)) + iteration},
        request_id="run",
        dependencies={},
    )
    assert successful.workflow_result.success is True
    assert successful.final_state["count"] == 3
    assert successful.terminated_reason == "condition_stopped"

    failed = loop_orchestration.run_workflow_loop(
        max_iterations=2,
        initial_state={"seed": 1},
        continue_predicate=lambda iteration, state: iteration <= 1,
        iteration_handler=lambda iteration, state: 123,  # type: ignore[return-value]
        request_id="run",
        dependencies={},
    )
    assert failed.workflow_result.success is False
    assert failed.terminated_reason == "iteration_failed"
    assert failed.final_state == {"seed": 1}

    assert loop_orchestration._parse_iteration(2) == 2
    assert loop_orchestration._parse_iteration("3") == 3
    assert loop_orchestration._parse_iteration("0") == 1
    assert loop_orchestration._parse_iteration("not-a-number") == 1


def test_multi_step_json_runtime_helpers_cover_summary_and_failure_paths() -> None:
    assert json_runtime.summarize_tool_action(tool_name="", tool_input={"x": 1}) == ""
    assert json_runtime.summarize_tool_action(tool_name="calc", tool_input=7).startswith("calc 7")
    long_tool_summary = json_runtime.summarize_tool_action(
        tool_name="calc",
        tool_input={"payload": "x" * 400},
    )
    assert long_tool_summary.endswith("...")
    assert len(long_tool_summary) == 320

    assert json_runtime.summarize_observation(final_output={}, error=" boom ") == "error: boom"
    assert json_runtime.summarize_observation(final_output={"x": 1}, error=None).startswith("{")
    long_observation = json_runtime.summarize_observation(final_output="x" * 500, error=None)
    assert long_observation.endswith("...")
    assert len(long_observation) == 320

    failure = json_runtime.build_json_final_result(
        final_state={
            "memory": [],
            "continuation_trace": [],
            "retrieval_trace": [],
            "memory_errors": [],
            "step_outputs": [],
            "tool_results": [ToolResult(tool_name="calculator", ok=False, error="boom")],
            "final_output": {},
            "terminated_reason": "failed",
            "last_model_response": LLMResponse(text="last"),
            "fatal_error": "fatal",
            "fatal_metadata": {"stage": "terminal"},
        },
        request_id="req",
        dependencies={"dep": object()},
        max_steps=3,
        stop_on_step_failure=True,
        alternatives_prompt_target="user",
        continuation_memory_tail_items=4,
        step_memory_tail_items=2,
        memory_namespace="default",
        memory_read_top_k=3,
        memory_write_observations=True,
        memory_store_enabled=True,
    )
    assert failure.success is False
    assert failure.output["error"] == "fatal"
    assert failure.metadata["stage"] == "terminal"


def test_planner_executor_helpers_cover_deserialization_and_callback_edge_cases() -> None:
    parsed = planner_helpers.deserialize_tool_results(
        [
            {"tool_name": "calculator", "ok": True, "result": {"value": 1}},
            "skip",
            {"tool_name": "missing-ok"},
        ]
    )
    assert len(parsed) == 1
    assert planner_helpers.deserialize_tool_results("not-a-list") == []

    assert planner_helpers.deserialize_model_response("bad") is None
    assert planner_helpers.deserialize_model_response({"provider": "x"}) is None

    callbacks = _planner_callbacks([{"step_id": "s1", "instruction": "do", "success_criteria": "done"}])
    with pytest.raises(ValueError, match="Loop metadata is required"):
        callbacks.executor_prompt_builder({})
    with pytest.raises(ValueError, match="out of bounds"):
        callbacks.executor_prompt_builder({"_loop": {"iteration": 2}, "loop_state": {}})

    bad_step_callbacks = _planner_callbacks([123])
    with pytest.raises(ValueError, match="not a mapping"):
        bad_step_callbacks.executor_prompt_builder({"_loop": {"iteration": 1}, "loop_state": {}})

    fallback_state = bad_step_callbacks.state_reducer(
        {},
        ExecutionResult(success=True),
        1,
    )
    assert fallback_state == {}

    missing_step_state = callbacks.state_reducer({}, ExecutionResult(success=True), 1)
    assert missing_step_state["step_results"][0]["error"].startswith("Workflow iteration did not include")

    iteration_step = WorkflowStepResult(
        step_id="execute_plan_step",
        status="completed",
        success=True,
        output={"success": "not-bool", "output": {"final_output": {"done": True}}},
        error=None,
    )
    reduced = callbacks.state_reducer(
        {},
        ExecutionResult(success=True, step_results={"execute_plan_step": iteration_step}),
        1,
    )
    assert reduced["step_results"][0]["success"] is True
