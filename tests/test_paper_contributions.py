"""Tests for deterministic, evidence-bounded paper contributions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

import design_research_agents as drag


class _DescribedLLM:
    def default_model(self) -> str:
        return "test-model"

    def describe(self) -> Mapping[str, object]:
        return {
            "provider": "test-provider",
            "model": "test-model",
            "api_key": "sk-super-secret-value",
            "headers": {"Authorization": "Bearer another-secret"},
            "base_url": "https://researcher:url-secret@example.test/v1",
        }


class _IncompleteLLM:
    pass


class _BrokenDescriptionLLM:
    def default_model(self) -> str:
        raise RuntimeError("not configured")

    def describe(self) -> Mapping[str, object]:
        raise RuntimeError("not configured")


def _callable_toolbox() -> drag.Toolbox:
    config = drag.CallableToolConfig(
        name="score_concept",
        description="Score one concept.",
        handler=lambda payload: {"score": payload.get("score", 0)},
        permissions=("read",),
    )
    return drag.Toolbox(enable_core_tools=False, callable_tools=(config,))


def _gap_ids(packet: Mapping[str, object]) -> set[str]:
    return {str(item["gap_id"]) for item in packet["reporting_gaps"]}  # type: ignore[index, union-attr]


def _contributions(packet: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = packet["contributions"]
    assert isinstance(value, list)
    assert all(isinstance(item, Mapping) for item in value)
    return value  # type: ignore[return-value]


def test_direct_call_emits_configured_methods_and_redacts_secrets() -> None:
    agent = drag.DirectLLMCall(
        llm_client=_DescribedLLM(),  # type: ignore[arg-type]
        system_prompt="Be concise.",
        temperature=0.2,
        max_tokens=40,
        provider_options={
            "api_token": "plain-secret",
            "note": "use Bearer third-secret",
        },
    )

    packet = drag.collect_agent_paper_contributions(agent)

    assert packet["schema_version"] == drag.PAPER_CONTRIBUTION_VERSION
    assert packet["source"]["component_id"] == "agents.direct-llm-call"
    assert _contributions(packet)[0]["evidence_basis"] == "configured"
    serialized = json.dumps(packet, sort_keys=True)
    assert "plain-secret" not in serialized
    assert "another-secret" not in serialized
    assert "third-secret" not in serialized
    assert "url-secret" not in serialized
    assert "[REDACTED]" in serialized
    assert "agent:agents.direct-llm-call:execution-not-provided" in _gap_ids(packet)


def test_observed_execution_reports_model_tools_failures_steps_and_trace() -> None:
    toolbox = _callable_toolbox()
    result = drag.ExecutionResult(
        success=False,
        model_response=drag.LLMResponse(
            text="candidate",
            model="test-model",
            provider="test-provider",
            finish_reason="stop",
            latency_ms=17,
        ),
        tool_results=[
            drag.ToolResult(tool_name="score_concept", ok=True, warnings=("rounded",)),
            drag.ToolResult(
                tool_name="missing_tool",
                ok=False,
                error={"type": "LookupError", "message": "Bearer do-not-leak"},
            ),
        ],
        step_results={
            "prepare": {"status": "completed"},
            "execute": {"status": "failed"},
        },
        execution_order=["prepare", "execute"],
        metadata={"trace_path": "evidence/run-001.jsonl"},
    )

    packet = drag.collect_agent_paper_contributions(
        toolbox,
        execution_result=result,
        evidence_refs=("evidence/run-001.json",),
    )

    observed = [item for item in _contributions(packet) if item["evidence_basis"] == "observed"]
    assert len(observed) == 5
    assert all(item["evidence_refs"] == ["evidence/run-001.json", "evidence/run-001.jsonl"] for item in observed)
    texts = [str(item["text"]) for item in observed]
    assert any("invoked tool 'score_concept'; the invocation succeeded" in text for text in texts)
    assert any("invoked tool 'missing_tool'; the invocation failed" in text for text in texts)
    assert "agent:agents.toolbox:tool-2-failed" in _gap_ids(packet)
    assert "do-not-leak" not in json.dumps(packet)


def test_execution_without_durable_evidence_never_emits_observed_claim() -> None:
    result = drag.ExecutionResult(success=True)
    packet = drag.collect_agent_paper_contributions(
        drag.Tracer(enabled=False),
        execution_result=result,
    )

    assert {item["evidence_basis"] for item in _contributions(packet)} == {"configured"}
    assert "agent:agents.tracer:execution-evidence-missing" in _gap_ids(packet)


def test_configured_tool_is_available_but_not_observed_as_invoked() -> None:
    toolbox = _callable_toolbox()

    packet = drag.collect_agent_paper_contributions(toolbox)

    methods = _contributions(packet)[0]
    configuration = methods["metadata"]["configuration"]  # type: ignore[index]
    assert configuration["available_tools"][0]["name"] == "score_concept"  # type: ignore[index]
    assert all(":tool:" not in str(item["contribution_id"]) for item in _contributions(packet))


@pytest.mark.parametrize(
    ("factory", "expected_id", "limit_key"),
    [
        (
            lambda llm, tools: drag.ProposeCriticPattern(llm_client=llm, tool_runtime=tools, max_iterations=2),
            "agents.pattern.propose-critic",
            "max_iterations",
        ),
        (
            lambda llm, tools: drag.DebatePattern(llm_client=llm, tool_runtime=tools, max_rounds=2),
            "agents.pattern.debate",
            "max_rounds",
        ),
        (
            lambda llm, tools: drag.PlanExecutePattern(llm_client=llm, tool_runtime=tools, max_iterations=2),
            "agents.pattern.plan-execute",
            "max_iterations",
        ),
    ],
)
def test_representative_patterns_emit_deterministic_configuration(
    factory: object,
    expected_id: str,
    limit_key: str,
) -> None:
    llm = _DescribedLLM()
    toolbox = _callable_toolbox()
    pattern = factory(llm, toolbox)  # type: ignore[operator]

    first = drag.collect_agent_paper_contributions(pattern)
    second = drag.collect_agent_paper_contributions(pattern)

    assert first == second
    assert first["source"]["component_id"] == expected_id
    configuration = _contributions(first)[0]["metadata"]["configuration"]  # type: ignore[index]
    assert configuration[limit_key] == 2  # type: ignore[index]
    assert configuration["available_tools"][0]["name"] == "score_concept"  # type: ignore[index]
    if expected_id == "agents.pattern.plan-execute":
        assert configuration["max_tool_calls_per_step"] == 5  # type: ignore[index]


@pytest.mark.parametrize("mode", ["direct", "json", "code"])
def test_multi_step_modes_report_limits_and_tool_availability(mode: str) -> None:
    toolbox = _callable_toolbox()
    agent = drag.MultiStepAgent(
        mode=mode,  # type: ignore[arg-type]
        llm_client=_DescribedLLM(),  # type: ignore[arg-type]
        tool_runtime=toolbox if mode != "direct" else None,
        max_steps=4,
        max_tool_calls_per_step=3,
        execution_timeout_seconds=7,
        allowed_tools=("score_concept",),
    )

    packet = drag.collect_agent_paper_contributions(agent)
    configuration = _contributions(packet)[0]["metadata"]["configuration"]  # type: ignore[index]

    assert configuration["mode"] == mode  # type: ignore[index]
    assert configuration["max_steps"] == 4  # type: ignore[index]
    if mode == "direct":
        assert configuration["available_tools"] == []  # type: ignore[index]
    else:
        assert configuration["available_tools"][0]["name"] == "score_concept"  # type: ignore[index]


def test_workflow_reports_step_topology_without_claiming_execution() -> None:
    nested = drag.LogicStep(step_id="inside", handler=lambda context: {})
    workflow = drag.Workflow(
        steps=(
            drag.LogicStep(step_id="prepare", handler=lambda context: {}),
            drag.ToolStep(step_id="score", tool_name="score_concept", dependencies=("prepare",)),
            drag.LoopStep(step_id="refine", steps=(nested,), dependencies=("score",), max_iterations=2),
        ),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        default_execution_mode="dag",
        default_failure_policy="propagate_failed_state",
    )

    packet = drag.collect_agent_paper_contributions(workflow)
    configuration = _contributions(packet)[0]["metadata"]["configuration"]  # type: ignore[index]
    steps = configuration["steps"]  # type: ignore[index]

    assert [step["step_id"] for step in steps] == ["prepare", "score", "refine"]
    assert steps[1]["configured_tool_name"] == "score_concept"
    assert steps[2]["nested_steps"][0]["step_id"] == "inside"


def test_model_selector_and_tracer_have_explicit_adapters(tmp_path: Path) -> None:
    selector_packet = drag.collect_agent_paper_contributions(drag.ModelSelector())
    selector_config = _contributions(selector_packet)[0]["metadata"]["configuration"]  # type: ignore[index]
    assert selector_config["candidate_count"] > 0  # type: ignore[index]
    assert selector_config["catalog_signature"]  # type: ignore[index]

    tracer_packet = drag.collect_agent_paper_contributions(
        drag.Tracer(trace_dir=tmp_path / "traces", enable_console=False),
        component_id="study.tracer",
    )
    assert tracer_packet["source"]["component_id"] == "study.tracer"
    assert "agent:study.tracer:trace-not-observed" in _gap_ids(tracer_packet)


def test_trace_path_alone_is_a_durable_evidence_reference() -> None:
    packet = drag.collect_agent_paper_contributions(
        drag.Tracer(enable_console=False),
        execution_result={
            "success": True,
            "metadata": {"trace_path": "traces/run.jsonl"},
            "execution_order": [],
            "step_results": {},
            "tool_results": [],
        },
    )
    observed = [item for item in _contributions(packet) if item["evidence_basis"] == "observed"]
    assert len(observed) == 2
    assert all(item["evidence_refs"] == ["traces/run.jsonl"] for item in observed)
    assert "agent:agents.tracer:trace-not-observed" not in _gap_ids(packet)


def test_mapping_tool_results_and_invalid_entries_are_normalized() -> None:
    packet = drag.collect_agent_paper_contributions(
        _callable_toolbox(),
        execution_result={
            "success": True,
            "tool_results": [
                {
                    "tool_name": "score_concept",
                    "ok": True,
                    "warnings": ["rounded"],
                    "artifacts": [{"path": "score.json"}],
                },
                object(),
            ],
            "model_response": {"model": "m", "provider": "p"},
            "step_results": {"one": object()},
            "execution_order": ["one"],
            "metadata": {},
        },
        evidence_refs=("evidence/run.json",),
    )
    assert "agent:agents.toolbox:tool-result-2-invalid" in _gap_ids(packet)
    assert any(":tool:1" in str(item["contribution_id"]) for item in _contributions(packet))


def test_custom_and_incomplete_clients_emit_actionable_gaps() -> None:
    custom_packet = drag.collect_agent_paper_contributions(object(), component_id="custom.component")
    assert custom_packet["source"]["component_id"] == "custom.component"
    assert "agent:custom.component:custom-component-metadata-incomplete" in _gap_ids(custom_packet)

    incomplete = drag.collect_agent_paper_contributions(
        drag.DirectLLMCall(llm_client=_IncompleteLLM()),  # type: ignore[arg-type]
    )
    assert "agent:agents.direct-llm-call:model-configuration-incomplete" in _gap_ids(incomplete)

    broken = drag.collect_agent_paper_contributions(
        drag.DirectLLMCall(llm_client=_BrokenDescriptionLLM()),  # type: ignore[arg-type]
    )
    assert "agent:agents.direct-llm-call:model-configuration-incomplete" in _gap_ids(broken)


def test_invalid_adapter_inputs_fail_clearly() -> None:
    tracer = drag.Tracer(enabled=False)
    with pytest.raises(ValueError, match="component_id"):
        drag.collect_agent_paper_contributions(tracer, component_id=" ")
    with pytest.raises(TypeError, match="execution_result"):
        drag.collect_agent_paper_contributions(tracer, execution_result=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evidence_refs"):
        drag.collect_agent_paper_contributions(
            tracer,
            execution_result=drag.ExecutionResult(success=True),
            evidence_refs="run.json",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="evidence_refs"):
        drag.collect_agent_paper_contributions(
            tracer,
            execution_result=drag.ExecutionResult(success=True),
            evidence_refs=("",),
        )
