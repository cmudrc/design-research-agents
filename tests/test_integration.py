from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from design_research_agents import PromptWorkflowAgent, SeededRandomBaselineAgent, integration, study
from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse, Usage
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


def test_integration_request_condition_and_resolution_edges() -> None:
    condition = integration.StudyCondition(
        condition_id=" cond ",
        label=" Label ",
        metadata={"group": "a"},
    )
    request = integration.AgentRunRequest(
        agent_ref="agent",
        prompt="prompt",
        request_id=" req ",
        dependencies={"condition": condition},
        agent_bindings={
            "agent": lambda current_condition: SimpleNamespace(run=lambda: {"text": current_condition.label})
        },
    )

    execution = integration.execute_agent_request(request)

    assert condition.condition_id == "cond"
    assert condition.label == "Label"
    assert request.request_id == "req"
    assert execution.output == {"text": "Label"}

    with pytest.raises(ValueError, match="condition_id"):
        integration.StudyCondition(condition_id=" ")
    with pytest.raises(ValueError, match="Unknown agent reference"):
        integration.execute_agent_run("MissingAgent", prompt="p", request_id=None, dependencies={})
    with pytest.raises(ValueError, match="not executable"):
        integration._invoke_agent(executable=object(), prompt="p", request_id=None, dependencies={})


def test_integration_callable_invocation_contract_edges() -> None:
    calls: list[tuple[str, object]] = []

    def accepts_public_contract(*, prompt: object, request_id: str | None, dependencies: dict[str, object]) -> str:
        calls.append(("public", (prompt, request_id, dependencies)))
        return "public"

    def accepts_input(*, input: object) -> str:
        calls.append(("input", input))
        return "input"

    def accepts_nothing() -> str:
        return "nothing"

    def accepts_one(value: object) -> str:
        calls.append(("one", value))
        return "one"

    def accepts_too_many(first: object, second: object) -> str:
        del first, second
        return "bad"

    assert integration._invoke_callable(
        callable_obj=accepts_public_contract,
        prompt="p",
        request_id="req",
        dependencies={"d": 1},
    ) == "public"
    assert integration._invoke_callable(
        callable_obj=accepts_input,
        prompt="p",
        request_id=None,
        dependencies={},
    ) == "input"
    assert integration._invoke_callable(
        callable_obj=accepts_nothing,
        prompt="p",
        request_id=None,
        dependencies={},
    ) == "nothing"
    assert integration._invoke_callable(callable_obj=accepts_one, prompt="p", request_id=None, dependencies={}) == "one"
    with pytest.raises(ValueError, match="public prompt"):
        integration._invoke_callable(callable_obj=accepts_too_many, prompt="p", request_id=None, dependencies={})

    assert calls[0][0] == "public"
    assert calls[-1] == ("one", "p")

    assert integration._is_condition_scoped_binding(object()) is False
    assert integration._is_condition_scoped_binding(lambda prompt: prompt) is False
    assert integration._is_condition_scoped_binding(lambda *args: args) is False
    assert integration._is_condition_scoped_binding(lambda condition: condition) is True


def test_integration_normalization_edges_cover_outputs_metrics_events_and_traces() -> None:
    mapping_execution = integration.normalize_agent_execution(
        {
            "outputs": {"value": 1},
            "metrics": {"score": 0.5},
            "trace_refs": ["trace-a", 3],
            "metadata": {"source": "mapping"},
            "events": [
                "skip",
                {
                    "event_type": "tool_call",
                    "text": "used tool",
                    "meta_json": "raw",
                    "tool_name": "calculator",
                    "run_id": "run-1",
                },
            ],
        },
        request_id="req",
    )
    assert mapping_execution.output == {"value": 1}
    assert mapping_execution.metrics == {"score": 0.5}
    assert mapping_execution.trace_refs == ["trace-a", "3"]
    assert mapping_execution.events[0]["meta_json"] == {"value": "raw"}
    assert mapping_execution.events[0]["tool_name"] == "calculator"

    text_execution = integration.normalize_agent_execution("plain", request_id="req")
    assert text_execution.output == {"text": "plain"}
    assert text_execution.events[0]["meta_json"] == {"auto_generated": True}

    result_execution = integration.normalize_agent_execution(
        ExecutionResult(
            success=True,
            output={
                "final_output": "done",
                "metrics": {"input_tokens": 99},
                "events": {"not": "a sequence"},
            },
            model_response=LLMResponse(
                text="done",
                model="m",
                provider="p",
                usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            ),
            metadata={"trace_path": "trace-a", "trace_refs": ["trace-a", "trace-b"]},
        ),
        request_id="req",
    )
    assert result_execution.output == {"text": "done"}
    assert result_execution.metrics == {"input_tokens": 99, "output_tokens": 2, "total_tokens": 3}
    assert result_execution.trace_refs == ["trace-a", "trace-b"]
    assert result_execution.metadata["model_name"] == "m"
    assert result_execution.metadata["model_provider"] == "p"

    object_usage_execution = integration.normalize_agent_execution(
        SimpleNamespace(
            success=True,
            output={"final_output": 7},
            metadata={},
            model_response=SimpleNamespace(usage={"prompt_tokens": 4}),
        ),
        request_id=None,
    )
    assert object_usage_execution.output == {"final_output": 7}
    assert object_usage_execution.metrics["input_tokens"] == 4
    assert integration._extract_execution_output({"model_text": "model says"}) == {"text": "model says"}
    assert integration._extract_output_mapping("bad") == {}
