from __future__ import annotations

from dataclasses import dataclass

import pytest

from design_research_agents import WorkflowStudyDelegate
from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents.workflow import LogicStep, Workflow


@dataclass(frozen=True)
class _RunSpec:
    run_id: str


@dataclass(frozen=True)
class _Condition:
    condition_id: str


def test_workflow_study_delegate_compile_exposes_wrapped_workflow() -> None:
    workflow = Workflow(steps=(LogicStep(step_id="emit", handler=lambda _context: {"final_output": {"ok": True}}),))
    delegate = WorkflowStudyDelegate(
        workflow=workflow,
        prompt_builder=lambda _problem_packet, _run_spec, _condition: "Study prompt.",
    )

    compiled = delegate.compile(
        prompt="ignored",
        dependencies={
            "problem_packet": object(),
            "run_spec": _RunSpec(run_id="run-1"),
            "condition": _Condition(condition_id="cond-1"),
        },
    )

    assert compiled.workflow is workflow
    assert compiled.input == "Study prompt."
    assert "emit" in compiled.to_mermaid(direction="LR")


def test_workflow_study_delegate_run_uses_prompt_builder_and_preserves_request_metadata() -> None:
    workflow = Workflow(steps=(LogicStep(step_id="emit", handler=lambda _context: {"final_output": {"ok": True}}),))
    captured: dict[str, object] = {}

    def _fake_run(
        input: str | dict[str, object] | None = None,
        *,
        execution_mode: str = "sequential",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: dict[str, object] | None = None,
    ) -> ExecutionResult:
        captured["input"] = input
        captured["execution_mode"] = execution_mode
        captured["failure_policy"] = failure_policy
        captured["request_id"] = request_id
        captured["dependencies"] = dependencies
        return ExecutionResult(
            success=True,
            output={"final_output": {"prompt": input}},
            metadata={"request_id": request_id or ""},
        )

    workflow.run = _fake_run  # type: ignore[method-assign]

    delegate = WorkflowStudyDelegate(
        workflow=workflow,
        prompt_builder=lambda problem_packet, run_spec, condition: (
            f"{getattr(problem_packet, 'brief', 'brief')}::{run_spec.run_id}::{condition.condition_id}"
        ),
    )
    problem_packet = type("ProblemPacket", (), {"brief": "Pick one design."})()
    run_spec = _RunSpec(run_id="run-7")
    condition = _Condition(condition_id="cond-a")

    result = delegate.run(
        prompt="ignored",
        request_id="study-run-7",
        dependencies={
            "problem_packet": problem_packet,
            "run_spec": run_spec,
            "condition": condition,
            "seed": 7,
        },
    )

    assert result.success is True
    assert captured["input"] == "Pick one design.::run-7::cond-a"
    assert captured["request_id"] == "study-run-7"
    assert captured["dependencies"] == {
        "problem_packet": problem_packet,
        "run_spec": run_spec,
        "condition": condition,
        "seed": 7,
    }


def test_workflow_study_delegate_accepts_direct_prompt_fallback() -> None:
    workflow = Workflow(steps=(LogicStep(step_id="emit", handler=lambda _context: {"final_output": {"ok": True}}),))
    captured: dict[str, object] = {}

    def _fake_run(
        input: str | dict[str, object] | None = None,
        *,
        execution_mode: str = "sequential",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: dict[str, object] | None = None,
    ) -> ExecutionResult:
        captured["input"] = input
        return ExecutionResult(success=True, output={"final_output": {"prompt": input}})

    workflow.run = _fake_run  # type: ignore[method-assign]
    delegate = WorkflowStudyDelegate(
        workflow=workflow,
        prompt_builder=lambda _problem_packet, _run_spec, _condition: "ignored",
    )

    delegate.run("Fallback direct prompt.")

    assert captured["input"] == "Fallback direct prompt."


def test_workflow_study_delegate_rejects_empty_prompt_sources() -> None:
    workflow = Workflow(steps=(LogicStep(step_id="emit", handler=lambda _context: {"final_output": {"ok": True}}),))
    delegate = WorkflowStudyDelegate(
        workflow=workflow,
        prompt_builder=lambda _problem_packet, _run_spec, _condition: "   ",
    )

    with pytest.raises(ValueError, match="empty prompt"):
        delegate.compile(
            prompt="ignored",
            dependencies={
                "problem_packet": object(),
                "run_spec": _RunSpec(run_id="run-1"),
                "condition": _Condition(condition_id="cond-1"),
            },
        )

    with pytest.raises(ValueError, match="non-empty prompt"):
        delegate.run(prompt="   ")
