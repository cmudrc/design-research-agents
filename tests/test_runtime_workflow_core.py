"""WorkflowRuntime execution tests for core step semantics."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from design_research_agents._contracts._workflow import (
    DelegateStep,
    LogicStep,
    LoopStep,
    ToolStep,
)
from design_research_agents._runtime._workflow._engine import WorkflowRuntime
from tests.helpers.workflow_stubs import (
    StaticMarkerAgent,
    StaticWorkflowDelegateRunner,
    StubToolRuntime,
)


def test_workflow_runtime_sequential_runs_dependency_order() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(step_id="a", handler=lambda ctx: {"value": 1}),
        LogicStep(
            step_id="b",
            dependencies=("a",),
            handler=lambda ctx: {
                "value": int(ctx["dependency_results"]["a"]["output"]["value"]) + 1,
            },
        ),
    ]

    result = workflow.run(steps, execution_mode="sequential")

    assert result.success
    assert result.execution_order == ["a", "b"]
    assert result.step_results["b"].output["value"] == 2


def test_workflow_runtime_loop_step_stops_when_continue_predicate_returns_false() -> None:
    workflow = WorkflowRuntime()
    result = workflow.run(
        [
            LoopStep(
                step_id="counter_loop",
                steps=(
                    LogicStep(
                        step_id="tick",
                        handler=lambda context: {
                            "counter": int(
                                (
                                    context.get("loop_state") if isinstance(context.get("loop_state"), Mapping) else {}
                                ).get("counter", 0)
                            )
                            + 1
                        },
                    ),
                ),
                max_iterations=5,
                initial_state={"counter": 0},
                continue_predicate=lambda iteration, state: int(state.get("counter", 0)) < 2,
                state_reducer=lambda state, iteration_result, iteration: {
                    "counter": int(iteration_result.step_results["tick"].output["counter"])
                },
                execution_mode="sequential",
                failure_policy="skip_dependents",
            )
        ],
        execution_mode="sequential",
    )
    loop_output = result.step_results["counter_loop"].output

    assert result.success
    assert loop_output["success"] is True
    assert loop_output["terminated_reason"] == "condition_stopped"
    assert loop_output["iterations"] == 5
    assert loop_output["iterations_executed"] == 2
    assert loop_output["final_state"]["counter"] == 2


def test_workflow_runtime_loop_step_terminates_at_max_iterations() -> None:
    workflow = WorkflowRuntime()
    result = workflow.run(
        [
            LoopStep(
                step_id="counter_loop",
                steps=(
                    LogicStep(
                        step_id="tick",
                        handler=lambda context: {
                            "counter": int(
                                (
                                    context.get("loop_state") if isinstance(context.get("loop_state"), Mapping) else {}
                                ).get("counter", 0)
                            )
                            + 1
                        },
                    ),
                ),
                max_iterations=3,
                initial_state={"counter": 0},
                state_reducer=lambda state, iteration_result, iteration: {
                    "counter": int(iteration_result.step_results["tick"].output["counter"])
                },
                execution_mode="sequential",
                failure_policy="skip_dependents",
            )
        ],
        execution_mode="sequential",
    )
    loop_output = result.step_results["counter_loop"].output

    assert result.success
    assert loop_output["success"] is True
    assert loop_output["terminated_reason"] == "max_iterations_reached"
    assert loop_output["iterations_executed"] == 3
    assert loop_output["final_state"]["counter"] == 3


def test_workflow_runtime_loop_step_propagates_iteration_failures() -> None:
    workflow = WorkflowRuntime()
    result = workflow.run(
        [
            LoopStep(
                step_id="counter_loop",
                steps=(
                    LogicStep(
                        step_id="tick",
                        handler=lambda context: (
                            (_ for _ in ()).throw(RuntimeError("boom"))
                            if int(
                                (context.get("_loop") if isinstance(context.get("_loop"), Mapping) else {}).get(
                                    "iteration", 0
                                )
                            )
                            == 2
                            else {"counter": 1}
                        ),
                    ),
                ),
                max_iterations=3,
                execution_mode="sequential",
                failure_policy="skip_dependents",
            )
        ],
        execution_mode="sequential",
    )
    loop_step = result.step_results["counter_loop"]
    loop_output = loop_step.output
    iteration_results = loop_output["iteration_results"]

    assert not result.success
    assert loop_step.status == "failed"
    assert loop_step.error == "Loop iteration failed."
    assert loop_output["success"] is False
    assert loop_output["terminated_reason"] == "iteration_failed"
    assert loop_output["iterations_executed"] == 2
    assert isinstance(iteration_results, list)
    assert iteration_results[1]["success"] is False


def test_workflow_runtime_loop_step_carries_state_across_iterations() -> None:
    workflow = WorkflowRuntime()
    result = workflow.run(
        [
            LoopStep(
                step_id="value_loop",
                steps=(
                    LogicStep(
                        step_id="double",
                        handler=lambda context: {
                            "value": int(
                                (
                                    context.get("loop_state") if isinstance(context.get("loop_state"), Mapping) else {}
                                ).get("value", 0)
                            )
                            * 2
                        },
                    ),
                ),
                max_iterations=4,
                initial_state={"value": 1},
                state_reducer=lambda state, iteration_result, iteration: {
                    "value": int(iteration_result.step_results["double"].output["value"])
                },
                execution_mode="sequential",
                failure_policy="skip_dependents",
            )
        ],
        execution_mode="sequential",
    )
    loop_output = result.step_results["value_loop"].output

    assert result.success
    assert loop_output["terminated_reason"] == "max_iterations_reached"
    assert loop_output["final_state"]["value"] == 16


@pytest.mark.parametrize(
    ("loop_step", "expected_error"),
    (
        (
            LoopStep(
                step_id="invalid_loop",
                steps=(LogicStep(step_id="body", handler=lambda context: {"ok": True}),),
                max_iterations=0,
            ),
            "LoopStep max_iterations must be >= 1.",
        ),
        (
            LoopStep(step_id="invalid_loop", steps=(), max_iterations=1),
            "LoopStep requires at least one nested step.",
        ),
    ),
)
def test_workflow_runtime_loop_step_rejects_invalid_configuration(
    loop_step: LoopStep,
    expected_error: str,
) -> None:
    workflow = WorkflowRuntime()

    result = workflow.run([loop_step], execution_mode="sequential")

    assert result.success is False
    loop_result = result.step_results["invalid_loop"]
    assert loop_result.status == "failed"
    assert loop_result.error == expected_error
    assert loop_result.metadata["stage"] == "loop_binding"


def test_workflow_runtime_loop_step_requires_mapping_reducer_output() -> None:
    workflow = WorkflowRuntime()

    result = workflow.run(
        [
            LoopStep(
                step_id="bad_reducer_loop",
                steps=(LogicStep(step_id="body", handler=lambda context: {"ok": True}),),
                max_iterations=1,
                state_reducer=lambda state, iteration_result, iteration: ["not", "a", "mapping"],
            )
        ],
        execution_mode="sequential",
    )

    loop_result = result.step_results["bad_reducer_loop"]
    assert result.success is False
    assert loop_result.status == "failed"
    assert loop_result.error == "LoopStep state_reducer must return a mapping."
    assert loop_result.metadata["stage"] == "loop_state_reducer"


@pytest.mark.parametrize(
    ("loop_step", "expected_error"),
    (
        (
            LoopStep(
                step_id="private_invalid_loop",
                steps=(LogicStep(step_id="body", handler=lambda context: {"ok": True}),),
                max_iterations=0,
            ),
            "LoopStep max_iterations must be >= 1.",
        ),
        (
            LoopStep(step_id="private_invalid_loop", steps=(), max_iterations=1),
            "LoopStep requires at least one nested step.",
        ),
    ),
)
def test_workflow_runtime_private_loop_executor_rejects_invalid_configuration(
    loop_step: LoopStep,
    expected_error: str,
) -> None:
    workflow = WorkflowRuntime()

    result = workflow._run_loop_step(
        step=loop_step,
        step_id="private_invalid_loop",
        step_context={},
        request_id="private-loop-test",
        dependencies={},
    )

    assert result.status == "failed"
    assert result.error == expected_error
    assert result.metadata["stage"] == "loop_binding"


def test_workflow_runtime_private_loop_executor_stops_before_first_iteration() -> None:
    workflow = WorkflowRuntime()

    result = workflow._run_loop_step(
        step=LoopStep(
            step_id="private_loop",
            steps=(LogicStep(step_id="body", handler=lambda context: {"ok": True}),),
            max_iterations=3,
            continue_predicate=lambda iteration, state: False,
        ),
        step_id="private_loop",
        step_context={},
        request_id="private-loop-test",
        dependencies={},
    )

    assert result.status == "completed"
    assert result.output["terminated_reason"] == "condition_stopped"
    assert result.output["iterations_executed"] == 0
    assert result.output["final_state"] == {}


def test_workflow_runtime_private_loop_executor_reduces_iteration_state() -> None:
    workflow = WorkflowRuntime()

    result = workflow._run_loop_step(
        step=LoopStep(
            step_id="private_loop",
            steps=(
                LogicStep(
                    step_id="tick",
                    handler=lambda context: {
                        "value": int(
                            (context.get("loop_state") if isinstance(context.get("loop_state"), Mapping) else {}).get(
                                "value",
                                0,
                            )
                        )
                        + 1
                    },
                ),
            ),
            max_iterations=2,
            initial_state={"value": 0},
            state_reducer=lambda state, iteration_result, iteration: {
                "value": iteration_result.step_results["tick"].output["value"],
                "iteration": iteration,
            },
            execution_mode="sequential",
        ),
        step_id="private_loop",
        step_context={"dependency_results": {"parent": {"output": {"value": "kept"}}}},
        request_id="private-loop-test",
        dependencies={"external": "dependency"},
    )

    assert result.status == "completed"
    assert result.output["terminated_reason"] == "max_iterations_reached"
    assert result.output["iterations_executed"] == 2
    assert result.output["final_state"] == {"value": 2, "iteration": 2}


def test_workflow_runtime_private_loop_executor_requires_mapping_reducer_output() -> None:
    workflow = WorkflowRuntime()

    result = workflow._run_loop_step(
        step=LoopStep(
            step_id="private_loop",
            steps=(LogicStep(step_id="body", handler=lambda context: {"ok": True}),),
            max_iterations=1,
            state_reducer=lambda state, iteration_result, iteration: ["not", "a", "mapping"],
        ),
        step_id="private_loop",
        step_context={},
        request_id="private-loop-test",
        dependencies={},
    )

    assert result.status == "failed"
    assert result.error == "LoopStep state_reducer must return a mapping."
    assert result.metadata["stage"] == "loop_state_reducer"


def test_workflow_runtime_private_loop_executor_reports_iteration_failure() -> None:
    workflow = WorkflowRuntime()

    result = workflow._run_loop_step(
        step=LoopStep(
            step_id="private_loop",
            steps=(
                LogicStep(
                    step_id="body",
                    handler=lambda context: (_ for _ in ()).throw(RuntimeError("iteration boom")),
                ),
            ),
            max_iterations=1,
        ),
        step_id="private_loop",
        step_context={},
        request_id="private-loop-test",
        dependencies={},
    )

    assert result.status == "failed"
    assert result.error == "Loop iteration failed."
    assert result.metadata["stage"] == "loop_execution"
    assert result.output["terminated_reason"] == "iteration_failed"
    assert result.output["iteration_results"][0]["step_results"]["body"]["error"] == "iteration boom"


def test_workflow_runtime_sequential_raises_for_unresolved_dependencies() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
        LogicStep(step_id="a", handler=lambda ctx: {"value": 1}),
    ]

    with pytest.raises(ValueError, match="cannot run before dependencies are resolved"):
        workflow.run(steps, execution_mode="sequential")


def test_workflow_runtime_dag_has_deterministic_topological_order() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(step_id="a", handler=lambda ctx: {"value": 1}),
        LogicStep(step_id="c", dependencies=("a",), handler=lambda ctx: {"value": 3}),
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
        LogicStep(
            step_id="d",
            dependencies=("b", "c"),
            handler=lambda ctx: {"value": 4},
        ),
    ]

    result = workflow.run(steps, execution_mode="dag")

    assert result.success
    assert result.execution_order == ["a", "b", "c", "d"]


def test_workflow_runtime_dag_detects_cycle_with_clear_error() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(step_id="a", dependencies=("b",), handler=lambda ctx: {"value": 1}),
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
    ]

    with pytest.raises(ValueError, match="Cycle detected"):
        workflow.run(steps, execution_mode="dag")


def test_workflow_runtime_failure_policy_skip_dependents() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(
            step_id="a",
            handler=lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
    ]

    result = workflow.run(
        steps,
        execution_mode="sequential",
        failure_policy="skip_dependents",
    )

    assert not result.success
    assert result.step_results["a"].status == "failed"
    assert result.step_results["b"].status == "skipped"
    assert result.step_results["b"].error == "skipped_upstream_failure"


def test_workflow_runtime_failure_policy_propagate_failed_state() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(
            step_id="a",
            handler=lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
    ]

    result = workflow.run(
        steps,
        execution_mode="sequential",
        failure_policy="propagate_failed_state",
    )

    assert not result.success
    assert result.step_results["a"].status == "failed"
    assert result.step_results["b"].status == "completed"


def test_workflow_runtime_route_branching_skips_non_selected_branch() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(
            step_id="router",
            handler=lambda ctx: {"route": "left"},
            route_map={"left": ("left_step",), "right": ("right_step",)},
        ),
        LogicStep(
            step_id="left_step",
            dependencies=("router",),
            handler=lambda ctx: {"value": 1},
        ),
        LogicStep(
            step_id="right_step",
            dependencies=("router",),
            handler=lambda ctx: {"value": 2},
        ),
    ]

    result = workflow.run(steps, execution_mode="dag", failure_policy="propagate_failed_state")

    assert result.success
    assert result.step_results["left_step"].status == "completed"
    assert result.step_results["right_step"].status == "skipped"
    assert result.step_results["right_step"].error == "skipped_branch_not_selected"


def test_workflow_runtime_route_map_error_fails_router_step() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(
            step_id="router",
            handler=lambda context: {"route": "missing"},
            route_map={"left": ("left_step",)},
        ),
        LogicStep(
            step_id="left_step",
            dependencies=("router",),
            handler=lambda context: {"value": 1},
        ),
    ]

    result = workflow.run(
        steps,
        execution_mode="sequential",
        failure_policy="skip_dependents",
    )

    assert result.success is False
    assert result.step_results["router"].status == "failed"
    assert result.step_results["router"].metadata["stage"] == "routing"
    assert "selected route 'missing'" in str(result.step_results["router"].error)
    assert result.step_results["left_step"].error == "skipped_upstream_failure"


def test_workflow_runtime_tool_step_returns_serialized_tool_result() -> None:
    tool_runtime = StubToolRuntime(
        handlers={"adder_tool": lambda payload: {"sum": float(payload.get("a", 0)) + float(payload.get("b", 0))}}
    )
    workflow = WorkflowRuntime(tool_runtime=tool_runtime)
    steps = [
        ToolStep(step_id="add", tool_name="adder_tool", input_data={"a": 40, "b": 2}),
    ]

    result = workflow.run(steps, execution_mode="sequential")

    assert result.success
    step_output = result.step_results["add"].output
    assert step_output["tool_name"] == "adder_tool"
    assert step_output["ok"] is True
    assert step_output["result"]["sum"] == 42.0


def test_workflow_runtime_agent_step_returns_serialized_agent_result() -> None:
    workflow = WorkflowRuntime()
    steps = [
        DelegateStep(
            step_id="delegate",
            delegate=StaticMarkerAgent(marker="math"),
            prompt="Solve this.",
        ),
    ]

    result = workflow.run(steps, execution_mode="sequential")

    assert result.success
    step_output = result.step_results["delegate"].output
    assert step_output["success"] is True
    assert step_output["output"]["agent_marker"] == "math"


def test_workflow_runtime_agent_step_accepts_nested_workflow_delegate() -> None:
    workflow = WorkflowRuntime()
    steps = [
        DelegateStep(
            step_id="delegate",
            delegate=StaticWorkflowDelegateRunner(),
            prompt="Route this prompt.",
        ),
    ]

    result = workflow.run(steps, execution_mode="sequential")

    assert result.success
    step_output = result.step_results["delegate"].output
    assert step_output["success"] is True
    assert step_output["step_results"]["nested_logic"]["output"]["prompt_echo"] == "Route this prompt."
    assert result.step_results["delegate"].metadata["delegate_type"] == "workflow"


def test_workflow_runtime_mixed_pipeline_supports_logic_agent_and_tool_steps() -> None:
    tool_runtime = StubToolRuntime(
        handlers={
            "text_length_tool": lambda payload: {
                "length": len(str(payload.get("text", ""))),
            }
        }
    )
    workflow = WorkflowRuntime(tool_runtime=tool_runtime)
    steps = [
        LogicStep(
            step_id="router",
            handler=lambda ctx: {"route": "agent_path"},
            route_map={"agent_path": ("delegate",), "other_path": ("unused",)},
        ),
        DelegateStep(
            step_id="delegate",
            delegate=StaticMarkerAgent(marker="proposal"),
            dependencies=("router",),
            prompt_builder=lambda ctx: "Write a proposal.",
        ),
        LogicStep(
            step_id="unused",
            dependencies=("router",),
            handler=lambda ctx: {"value": "should skip"},
        ),
        ToolStep(
            step_id="measure",
            tool_name="text_length_tool",
            dependencies=("delegate",),
            input_builder=lambda ctx: {
                "text": ctx["dependency_results"]["delegate"]["output"]["output"]["agent_marker"]
            },
        ),
        LogicStep(
            step_id="finalize",
            dependencies=("measure",),
            handler=lambda ctx: {"length": ctx["dependency_results"]["measure"]["output"]["result"]["length"]},
        ),
    ]

    result = workflow.run(steps, execution_mode="dag")

    assert result.success
    assert result.step_results["unused"].status == "skipped"
    assert result.step_results["unused"].error == "skipped_branch_not_selected"
    assert result.step_results["finalize"].output["length"] == len("proposal")


def test_workflow_runtime_unknown_bindings_fail_with_stage_metadata() -> None:
    tool_runtime = StubToolRuntime(handlers={"known_tool": lambda payload: {"ok": True}})
    workflow = WorkflowRuntime(tool_runtime=tool_runtime)
    steps = [
        ToolStep(step_id="missing_tool", tool_name="unknown_tool"),
        DelegateStep(step_id="missing_agent", delegate=object(), prompt="Do work."),
    ]

    result = workflow.run(
        steps,
        execution_mode="sequential",
        failure_policy="propagate_failed_state",
    )

    assert not result.success
    assert result.step_results["missing_tool"].status == "failed"
    assert result.step_results["missing_tool"].metadata["stage"] == "tool_binding"
    assert result.step_results["missing_agent"].status == "failed"
    assert result.step_results["missing_agent"].metadata["stage"] == "execution"
