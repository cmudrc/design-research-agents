"""Focused tests for the generic ``Workflow`` facade."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from design_research_agents._contracts._llm import (
    LLMChatParams,
    LLMMessage,
    LLMResponse,
)
from design_research_agents._contracts._workflow import (
    DelegateStep,
    LogicStep,
    LoopStep,
    ToolStep,
)
from design_research_agents._schemas import SchemaValidationError
from design_research_agents.patterns import DebatePattern
from design_research_agents.tools import Toolbox
from design_research_agents.workflow import Workflow, list_of, scalar, typed_dict
from tests.helpers.workflow_stubs import CaptureDependenciesAgent, StaticJsonDraftAgent


def _write_dataset(*, base_dir: str, filename: str) -> str:
    path = Path(base_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "participant_id,study_arm,satisfaction_score,notes",
                "P001,A,4.2,Onboarding clear",
                "P002,A,3.8,",
                "P003,B,,Needed more examples",
                "P004,B,4.9,Very helpful",
                "P005,A,2.7,Confusing navigation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _pure_dataset_steps() -> list[LogicStep | ToolStep]:
    return [
        ToolStep(
            step_id="describe_dataset",
            tool_name="data.describe",
            input_builder=lambda context: {
                "path": context["inputs"]["dataset_csv_path"],
                "kind": "csv",
            },
        ),
        ToolStep(
            step_id="load_sample",
            tool_name="data.load_csv",
            dependencies=("describe_dataset",),
            input_builder=lambda context: {
                "path": context["inputs"]["dataset_csv_path"],
                "nrows": context["inputs"]["sample_nrows"],
            },
        ),
        LogicStep(
            step_id="quality_gate",
            dependencies=("describe_dataset", "load_sample"),
            handler=lambda context: {
                "rows": context["dependency_results"]["describe_dataset"]["output"]["result"]["rows"],
                "sample_count": context["dependency_results"]["load_sample"]["output"]["result"]["count"],
                "threshold": context["inputs"]["max_missing_ratio_per_column"],
                "required_columns": context["inputs"]["required_columns"],
            },
        ),
        ToolStep(
            step_id="persist_report",
            tool_name="fs.write_text",
            dependencies=("quality_gate",),
            input_builder=lambda context: {
                "path": context["inputs"]["quality_report_path"],
                "content": json.dumps(
                    context["dependency_results"]["quality_gate"]["output"],
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                "overwrite": True,
            },
        ),
        LogicStep(
            step_id="finalize",
            dependencies=("persist_report",),
            handler=lambda context: {
                "report_path": context["dependency_results"]["persist_report"]["output"]["result"]["path"]
            },
        ),
    ]


def _pure_input_schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": [
            "dataset_csv_path",
            "quality_report_path",
            "required_columns",
            "sample_nrows",
            "max_missing_ratio_per_column",
        ],
        "properties": {
            "dataset_csv_path": {"type": "string"},
            "quality_report_path": {"type": "string"},
            "required_columns": {"type": "array", "items": {"type": "string"}},
            "sample_nrows": {"type": "integer"},
            "max_missing_ratio_per_column": {"type": "number"},
        },
        "additionalProperties": False,
    }


def _mixed_branching_steps(*, delegate: object) -> list[LogicStep | DelegateStep | ToolStep]:
    return [
        LogicStep(
            step_id="router",
            handler=lambda context: {
                "route": ("template_path" if str(context["prompt"]).lower().startswith("template:") else "agent_path")
            },
            route_map={
                "agent_path": ("draft_agent",),
                "template_path": ("draft_template",),
            },
        ),
        DelegateStep(
            step_id="draft_agent",
            delegate=delegate,
            dependencies=("router",),
            prompt_builder=lambda context: str(context["prompt"]),
        ),
        ToolStep(
            step_id="parse_agent_json",
            tool_name="text.extract_json",
            dependencies=("draft_agent",),
            input_builder=lambda context: {
                "text": context["dependency_results"]["draft_agent"]["output"]["output"]["model_text"]
            },
        ),
        LogicStep(
            step_id="finalize_agent",
            dependencies=("parse_agent_json",),
            handler=lambda context: {
                "branch": "agent",
                "title": context["dependency_results"]["parse_agent_json"]["output"]["result"]["json"].get("title", ""),
            },
        ),
        LogicStep(
            step_id="draft_template",
            dependencies=("router",),
            handler=lambda context: {
                "title": "Template fallback",
                "summary": f"Fallback summary for: {context['prompt']}",
            },
        ),
        LogicStep(
            step_id="finalize_template",
            dependencies=("draft_template",),
            handler=lambda context: {
                "branch": "template",
                "title": context["dependency_results"]["draft_template"]["output"]["title"],
            },
        ),
    ]


def test_workflow_schema_mode_accepts_user_defined_steps_with_inputs(tmp_path: Path) -> None:
    dataset_path = _write_dataset(base_dir=str(tmp_path), filename="pure_arbitrary_dataset.csv")
    report_path = tmp_path / "artifacts" / "pure_arbitrary_report.json"
    workflow = Workflow(
        tool_runtime=Toolbox(workspace_root=tmp_path),
        steps=_pure_dataset_steps(),
        input_schema=_pure_input_schema(),
    )

    result = workflow.run(
        {
            "dataset_csv_path": dataset_path,
            "quality_report_path": str(report_path),
            "required_columns": ["participant_id", "study_arm"],
            "sample_nrows": 3,
            "max_missing_ratio_per_column": 0.3,
        },
        request_id="test-pure-arbitrary",
    )

    assert result.success
    assert result.step_results["finalize"].status == "completed"
    assert str(result.step_results["finalize"].output["report_path"]) == str(report_path)


def test_workflow_schema_mode_validates_inputs_with_schema_hook(tmp_path: Path) -> None:
    dataset_path = _write_dataset(base_dir=str(tmp_path), filename="pure_schema_dataset.csv")
    workflow = Workflow(
        tool_runtime=Toolbox(workspace_root=tmp_path),
        steps=_pure_dataset_steps(),
        input_schema=_pure_input_schema(),
    )

    with pytest.raises(SchemaValidationError):
        workflow.run(
            {
                "dataset_csv_path": dataset_path,
                "quality_report_path": str(tmp_path / "artifacts" / "pure_schema_report.json"),
                "required_columns": ["participant_id"],
                "sample_nrows": "3",
                "max_missing_ratio_per_column": 0.2,
            }
        )


def test_workflow_schema_mode_without_schema_allows_arbitrary_inputs() -> None:
    workflow = Workflow(
        tool_runtime=Toolbox(),
        steps=[
            LogicStep(
                step_id="echo_inputs",
                handler=lambda context: {"inputs_snapshot": dict(context["inputs"])},
            )
        ],
        input_schema={},
    )

    result = workflow.run({"free_form": {"a": 1, "b": 2}}, request_id="test-pure-free")

    assert result.success
    assert result.step_results["echo_inputs"].output["inputs_snapshot"]["free_form"]["a"] == 1


def test_workflow_requires_non_empty_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        Workflow(
            tool_runtime=Toolbox(),
            steps=[],
        )


def test_workflow_prompt_mode_executes_user_defined_branching_steps() -> None:
    writer_agent = StaticJsonDraftAgent(payload={"title": "Agent title", "summary": "Agent summary"})
    workflow = Workflow(
        tool_runtime=Toolbox(),
        steps=_mixed_branching_steps(delegate=writer_agent),
        base_context={"audience": "research"},
    )

    agent_result = workflow.run("Write a concise brief.", request_id="test-mixed-agent")
    template_result = workflow.run(
        "template: Use deterministic fallback.",
        request_id="test-mixed-template",
    )

    assert agent_result.success
    assert agent_result.step_results["finalize_agent"].status == "completed"
    assert template_result.success
    assert template_result.step_results["finalize_template"].status == "completed"
    assert writer_agent.run_count == 1


def test_workflow_prompt_mode_injects_prompt_and_preserves_base_context() -> None:
    writer_agent = StaticJsonDraftAgent(payload={"title": "ignored"})
    custom_steps = [
        DelegateStep(
            step_id="delegate",
            delegate=writer_agent,
            prompt_builder=lambda context: f"{context['base_tag']}::{context['prompt']}",
        ),
        LogicStep(
            step_id="finalize",
            dependencies=("delegate",),
            handler=lambda context: {
                "base_tag": context["base_tag"],
                "prompt_seen": context["prompt"],
                "model_text": context["dependency_results"]["delegate"]["output"]["output"]["model_text"],
            },
        ),
    ]
    workflow = Workflow(
        tool_runtime=Toolbox(),
        steps=custom_steps,
        base_context={"base_tag": "custom"},
    )

    result = workflow.run(
        "Produce a short custom mixed-workflow brief.",
        request_id="test-mixed-context-injection",
    )

    assert result.success
    assert result.step_results["finalize"].output["base_tag"] == "custom"
    assert result.step_results["finalize"].output["prompt_seen"] == ("Produce a short custom mixed-workflow brief.")


def test_workflow_allows_agent_steps_nested_inside_loop_step() -> None:
    writer_agent = StaticJsonDraftAgent(payload={"title": "loop title"})
    workflow = Workflow(
        tool_runtime=Toolbox(),
        steps=[
            LoopStep(
                step_id="agent_loop",
                steps=(
                    DelegateStep(
                        step_id="delegate",
                        delegate=writer_agent,
                        prompt_builder=lambda context: str(context["prompt"]),
                    ),
                ),
                max_iterations=1,
                execution_mode="sequential",
            )
        ],
    )

    result = workflow.run("Draft a loop response.")

    assert result.success
    assert writer_agent.run_count == 1


def test_workflow_prompt_mode_default_run_controls_and_dependencies_are_applied() -> None:
    capture_agent = CaptureDependenciesAgent()
    workflow = Workflow(
        tool_runtime=Toolbox(),
        steps=[
            DelegateStep(step_id="delegate", delegate=capture_agent, prompt="Run"),
            LogicStep(
                step_id="finalize",
                dependencies=("delegate",),
                handler=lambda context: {
                    "workflow": dict(context["_workflow"]),
                },
            ),
        ],
        default_execution_mode="sequential",
        default_failure_policy="propagate_failed_state",
        default_request_id_prefix="mixed-default",
        default_dependencies={"from_default": "yes"},
    )

    result = workflow.run(
        "irrelevant prompt",
        dependencies={"from_run": "yes"},
    )
    workflow_meta = result.step_results["finalize"].output["workflow"]
    assert str(workflow_meta["request_id"]).startswith("mixed-default:")
    assert workflow_meta["execution_mode"] == "sequential"
    assert workflow_meta["failure_policy"] == "propagate_failed_state"
    assert capture_agent.last_dependencies is not None
    assert capture_agent.last_dependencies["from_default"] == "yes"
    assert capture_agent.last_dependencies["from_run"] == "yes"


def test_workflow_schema_mode_default_run_controls_are_applied() -> None:
    workflow = Workflow(
        tool_runtime=Toolbox(),
        input_schema={},
        steps=[
            LogicStep(
                step_id="inspect",
                handler=lambda context: {"workflow": dict(context["_workflow"])},
            )
        ],
        default_execution_mode="dag",
        default_failure_policy="propagate_failed_state",
        default_request_id_prefix="pure-default",
    )

    result = workflow.run({})
    workflow_meta = result.step_results["inspect"].output["workflow"]
    assert str(workflow_meta["request_id"]).startswith("pure-default:")
    assert workflow_meta["execution_mode"] == "dag"
    assert workflow_meta["failure_policy"] == "propagate_failed_state"


def test_workflow_infers_prompt_mode_when_input_schema_is_omitted() -> None:
    workflow = Workflow(
        tool_runtime=Toolbox(),
        steps=[LogicStep(step_id="echo", handler=lambda context: {"prompt": context["prompt"]})],
    )

    result = workflow.run("hello")
    assert result.success
    assert result.step_results["echo"].output["prompt"] == "hello"

    with pytest.raises(ValueError, match="without input_schema"):
        workflow.run({"not": "prompt"})


def test_workflow_infers_schema_mode_when_input_schema_is_provided() -> None:
    workflow = Workflow(
        tool_runtime=Toolbox(),
        input_schema={},
        steps=[
            LogicStep(
                step_id="echo_inputs",
                handler=lambda context: {"inputs_snapshot": dict(context["inputs"])},
            )
        ],
    )

    result = workflow.run({"k": 1})
    assert result.success
    assert result.step_results["echo_inputs"].output["inputs_snapshot"]["k"] == 1

    with pytest.raises(ValueError, match="with input_schema"):
        workflow.run("prompt input")


def test_workflow_output_schema_validation_supports_helpers() -> None:
    output_schema = list_of(
        typed_dict(
            required={"id": scalar("integer")},
            optional={"label": scalar("string")},
        )
    )
    workflow = Workflow(
        tool_runtime=Toolbox(),
        steps=[
            LogicStep(
                step_id="emit",
                handler=lambda _context: {
                    "final_output": [{"id": 1, "label": "alpha"}, {"id": 2}],
                },
            )
        ],
        output_schema=output_schema,
    )

    result = workflow.run("emit rows")
    assert result.success
    assert result.output["final_output"] == [{"id": 1, "label": "alpha"}, {"id": 2}]


def test_workflow_output_schema_validation_raises_on_invalid_payload() -> None:
    workflow = Workflow(
        tool_runtime=Toolbox(),
        steps=[
            LogicStep(
                step_id="emit",
                handler=lambda _context: {"final_output": [{"id": "not-int"}]},
            )
        ],
        output_schema=list_of(typed_dict(required={"id": scalar("integer")})),
    )

    with pytest.raises(SchemaValidationError, match="output\\.final_output\\[0\\]\\.id"):
        workflow.run("emit invalid rows")


def test_debate_pattern_runs_rounds_and_returns_judged_verdict() -> None:
    class _SequenceLLMClient:
        def __init__(self, responses: Sequence[str]) -> None:
            self._responses = list(responses)

        def chat(
            self,
            messages: Sequence[LLMMessage],
            *,
            model: str,
            params: LLMChatParams,
        ) -> LLMResponse:
            del messages, model, params
            if not self._responses:
                raise AssertionError("No more stubbed responses available.")
            return LLMResponse(model="noop-model", text=self._responses.pop(0), provider="noop")

        def default_model(self) -> str:
            return "noop-model"

        def close(self) -> None:
            return None

        def __enter__(self) -> _SequenceLLMClient:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            self.close()
            return None

    workflow = DebatePattern(
        llm_client=_SequenceLLMClient(
            [
                "Affirmative argument",
                "Negative argument",
                json.dumps(
                    {
                        "winner": "affirmative",
                        "rationale": "Affirmative provided stronger evidence.",
                        "synthesis": "Adopt a phased rollout with explicit safeguards.",
                    }
                ),
            ]
        ),
        tool_runtime=Toolbox(),
        max_rounds=1,
    )

    result = workflow.run("Should the team launch this product now?")

    assert result.success
    assert workflow.workflow is not None
    assert result.output["final_output"]["winner"] == "affirmative"
    assert result.output["terminated_reason"] == "completed"
    assert result.output["details"]["verdict"]["synthesis"].startswith("Adopt a phased rollout")
    assert str(result.metadata["request_id"]).startswith("debate:")
