from __future__ import annotations

from dataclasses import dataclass

from design_research_agents import PromptWorkflowAgent, SeededRandomBaselineAgent, integration, study
from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents.workflow import LogicStep, Workflow


@dataclass(frozen=True)
class _RunSpec:
    run_id: str
    seed: int


@dataclass(frozen=True)
class _Condition:
    condition_id: str


@dataclass(frozen=True)
class _ProblemMetadata:
    problem_id: str


class _DecisionProblem:
    def __init__(self) -> None:
        self.metadata = _ProblemMetadata(problem_id="stub-decision")
        self.prompt = "Choose one cooling-fin design."
        self._candidates = (
            {"fan_count": 4.0, "fin_gap_mm": 2.5},
            {"fan_count": 6.0, "fin_gap_mm": 2.0},
            {"fan_count": 8.0, "fin_gap_mm": 1.5},
        )

    def iter_candidates(self) -> tuple[dict[str, float], ...]:
        return self._candidates


def test_public_seeded_random_agent_executes_through_integration() -> None:
    problem = _DecisionProblem()
    run_spec = _RunSpec(run_id="run-1", seed=7)
    condition = _Condition(condition_id="cond-a")

    execution = integration.execute_agent_run(
        "SeededRandomBaselineAgent",
        prompt=problem,
        request_id=run_spec.run_id,
        dependencies={
            "problem": problem,
            "run_spec": run_spec,
            "condition": condition,
            "seed": run_spec.seed,
        },
    )

    assert execution.output
    assert execution.metadata["problem_id"] == "stub-decision"
    assert execution.metadata["request_id"] == "run-1"
    assert execution.events[0]["event_type"]


def test_public_study_request_executes_through_typed_facade() -> None:
    problem = _DecisionProblem()
    run_request = study.AgentRunRequest(
        agent_ref="SeededRandomBaselineAgent",
        prompt=problem,
        request_id="run-request",
        dependencies={
            "problem": problem,
            "run_spec": _RunSpec(run_id="run-request", seed=5),
            "condition": study.StudyCondition(condition_id="cond-request"),
            "seed": 5,
        },
    )

    execution = study.execute_agent_request(run_request)

    assert execution.output
    assert execution.metadata["request_id"] == "run-request"
    assert execution.metadata["condition_id"] == "cond-request"


def test_public_normalize_agent_execution_reuses_owner_envelope() -> None:
    execution = integration.normalize_agent_execution(
        {
            "text": "done",
            "metrics": {"primary_outcome": 1.0},
            "events": [{"event_type": "assistant_output", "text": "done"}],
        },
        request_id="run-normalize",
    )

    assert execution.output == {"text": "done"}
    assert execution.metrics["primary_outcome"] == 1.0
    assert execution.events[0]["session_id"] == "run-normalize"


def test_seeded_random_and_prompt_workflow_normalize_to_same_envelope_shape() -> None:
    problem_packet = type("ProblemPacket", (), {"problem_id": "problem-1", "brief": "Pick one design."})()
    run_spec = _RunSpec(run_id="run-2", seed=11)
    condition = _Condition(condition_id="cond-b")

    workflow = Workflow(steps=(LogicStep(step_id="emit", handler=lambda _context: {"final_output": {"ok": True}}),))

    def _fake_run(
        input: str | dict[str, object] | None = None,
        *,
        execution_mode: str = "sequential",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: dict[str, object] | None = None,
    ) -> ExecutionResult:
        del execution_mode, failure_policy, dependencies
        return ExecutionResult(
            success=True,
            output={
                "final_output": {"prompt": input},
                "metrics": {"primary_outcome": 1.0},
                "events": [
                    {
                        "event_type": "assistant_output",
                        "text": str(input),
                        "actor_id": "agent",
                    }
                ],
            },
            metadata={"request_id": request_id or "", "trace_path": "traces/run-2.jsonl"},
        )

    workflow.run = _fake_run  # type: ignore[method-assign]
    prompt_agent = PromptWorkflowAgent(
        workflow=workflow,
        prompt_builder=lambda packet, spec, current_condition: (
            f"{packet.problem_id}:{spec.run_id}:{current_condition.condition_id}"
        ),
    )

    prompt_execution = integration.execute_agent_run(
        "prompt-agent",
        prompt="ignored",
        request_id=run_spec.run_id,
        dependencies={
            "problem_packet": problem_packet,
            "run_spec": run_spec,
            "condition": condition,
            "seed": run_spec.seed,
        },
        agent_bindings={"prompt-agent": prompt_agent},
    )
    baseline_execution = integration.execute_agent_run(
        SeededRandomBaselineAgent(seed=run_spec.seed),
        prompt=_DecisionProblem(),
        request_id="run-3",
        dependencies={
            "problem": _DecisionProblem(),
            "run_spec": _RunSpec(run_id="run-3", seed=run_spec.seed),
            "condition": condition,
            "seed": run_spec.seed,
        },
    )

    assert prompt_execution.output
    assert prompt_execution.metrics
    assert prompt_execution.events
    assert isinstance(prompt_execution.trace_refs, list)
    assert prompt_execution.metadata["request_id"] == "run-2"

    assert baseline_execution.output
    assert isinstance(baseline_execution.metrics, dict)
    assert baseline_execution.events
    assert isinstance(baseline_execution.trace_refs, list)
    assert baseline_execution.metadata["request_id"] == "run-3"
