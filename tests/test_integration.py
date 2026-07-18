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


def test_study_condition_and_request_normalize_caller_owned_values() -> None:
    metadata = {"group": "a"}
    bindings = {"agent": lambda: "done"}
    condition = integration.StudyCondition(condition_id=" condition ", label=" ", metadata=metadata)
    request = integration.AgentRunRequest(
        agent_ref="agent",
        prompt="prompt",
        request_id=" ",
        dependencies=metadata,
        agent_bindings=bindings,
    )

    metadata["group"] = "changed"
    bindings.clear()
    assert condition.condition_id == "condition"
    assert condition.label is None
    assert condition.metadata == {"group": "a"}
    assert request.request_id is None
    assert request.dependencies == {"group": "a"}
    assert "agent" in request.agent_bindings

    with pytest.raises(ValueError, match="condition_id must be non-empty"):
        integration.StudyCondition(condition_id=" ")


def test_agent_reference_and_binding_resolution_rejects_ambiguous_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Unknown agent reference"):
        integration._resolve_agent_ref("MissingAgent", condition=None, agent_bindings=None)

    exported = object()
    monkeypatch.setattr(integration, "import_module", lambda _name: SimpleNamespace(ExportedAgent=exported))
    assert integration._resolve_agent_ref("ExportedAgent", condition=None, agent_bindings=None) is exported

    run_object = SimpleNamespace(run=lambda *, prompt: prompt)
    assert not integration._is_condition_scoped_binding(run_object)
    assert not integration._is_condition_scoped_binding("not callable")
    assert not integration._is_condition_scoped_binding(lambda prompt: prompt)
    assert not integration._is_condition_scoped_binding(lambda *args: args)
    assert not integration._is_condition_scoped_binding(lambda left, right: (left, right))
    monkeypatch.setattr(integration.inspect, "signature", lambda _binding: (_ for _ in ()).throw(ValueError("bad")))
    assert not integration._is_condition_scoped_binding(lambda value: value)


def test_agent_invocation_supports_all_documented_callable_shapes() -> None:
    assert integration._invoke_agent(
        executable=lambda *, prompt, request_id, dependencies: (prompt, request_id, dependencies),
        prompt="hello",
        request_id="request",
        dependencies={"value": 1},
    ) == ("hello", "request", {"value": 1})
    assert (
        integration._invoke_callable(
            callable_obj=lambda *, input: input,
            prompt="hello",
            request_id=None,
            dependencies={},
        )
        == "hello"
    )
    assert (
        integration._invoke_callable(
            callable_obj=lambda: "no-args",
            prompt="ignored",
            request_id=None,
            dependencies={},
        )
        == "no-args"
    )
    assert (
        integration._invoke_callable(
            callable_obj=lambda value, /: value,
            prompt="positional",
            request_id=None,
            dependencies={},
        )
        == "positional"
    )

    with pytest.raises(ValueError, match="not executable"):
        integration._invoke_agent(executable=object(), prompt="hello", request_id=None, dependencies={})
    with pytest.raises(ValueError, match="must accept the public"):
        integration._invoke_callable(
            callable_obj=lambda left, right: (left, right),
            prompt="hello",
            request_id=None,
            dependencies={},
        )


def test_execution_normalization_covers_mapping_scalar_and_execution_result_shapes() -> None:
    mapping_envelope = integration.normalize_agent_execution(
        {
            "outputs": {"value": 2},
            "trace_refs": [3, "trace.jsonl"],
            "metadata": {"source": "mapping"},
            "events": [
                "invalid",
                {
                    "event_type": "tool",
                    "meta_json": "raw",
                    "level": "step",
                    "tool_name": "search",
                },
            ],
        },
        request_id="request",
    )
    scalar_envelope = integration.normalize_agent_execution(42, request_id="scalar")

    assert mapping_envelope.output == {"value": 2}
    assert mapping_envelope.trace_refs == ["3", "trace.jsonl"]
    assert mapping_envelope.events[0]["meta_json"] == {"value": "raw"}
    assert mapping_envelope.events[0]["tool_name"] == "search"
    assert scalar_envelope.output == {"text": "42"}
    assert scalar_envelope.events[0]["meta_json"] == {"auto_generated": True}

    result = ExecutionResult(
        success=True,
        output={"final_output": "finished", "metrics": {"input_tokens": 99}, "events": "invalid"},
        model_response=LLMResponse(
            text="finished",
            model=" model ",
            provider=" provider ",
            usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        ),
        metadata={"trace_path": "trace.jsonl", "trace_refs": ["trace.jsonl", "second.jsonl", 3]},
    )
    result_envelope = integration.normalize_agent_execution(result, request_id="execution")
    assert result_envelope.output == {"text": "finished"}
    assert result_envelope.metrics == {"input_tokens": 99, "output_tokens": 3, "total_tokens": 5}
    assert result_envelope.metadata["model_name"] == " model "
    assert result_envelope.metadata["model_provider"] == " provider "
    assert result_envelope.trace_refs == ["trace.jsonl", "second.jsonl"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"final_output": {"value": 1}}, {"value": 1}),
        ({"final_output": 3}, {"final_output": 3}),
        ({"text": 4}, {"text": "4"}),
        ({"model_text": 5}, {"text": "5"}),
        ({"value": 6}, {"value": 6}),
    ],
)
def test_execution_output_normalization_variants(payload: dict[str, object], expected: dict[str, object]) -> None:
    assert integration._extract_execution_output(payload) == expected


def test_usage_mapping_and_non_mapping_execution_fields_are_normalized() -> None:
    metrics: dict[str, object] = {}
    integration._merge_usage_metrics(
        metrics,
        SimpleNamespace(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
    )
    assert metrics == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    assert integration._extract_output_mapping("invalid") == {}

    execution_like = SimpleNamespace(
        success=True,
        output="invalid",
        metadata="invalid",
        model_response=None,
    )
    envelope = integration.normalize_agent_execution(execution_like)
    assert envelope.output == {}
    assert envelope.metadata == {}
