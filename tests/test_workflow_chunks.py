"""Focused tests for generic pure and mixed workflow chunk facades."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from design_research_agents.contracts.agent import Agent, AgentResult
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents.contracts.workflow import AgentStep, LogicStep, ToolStep
from design_research_agents.schemas import SchemaValidationError
from design_research_agents.tools import UnifiedToolRuntime
from design_research_agents.workflow.implementations.mixed_agent_workflow import (
    mixed_agent_workflow,
)
from design_research_agents.workflow.implementations.pure_tool_workflow import (
    pure_tool_workflow,
)


class _NoopLLMClient:
    """LLM stub used when tests inject concrete agents directly."""

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        return LLMResponse(model=model, text="{}", provider="noop")

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.chat(
            request.messages,
            model=request.model or self.default_model(),
            params=LLMChatParams(),
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        return "noop-model"


class _StaticJsonDraftAgent(Agent):
    """Agent stub that always returns one JSON object in ``output.model_text``."""

    def __init__(self, *, payload: Mapping[str, object]) -> None:
        self._payload = dict(payload)
        self.run_count = 0

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del prompt, request_id, dependencies
        self.run_count += 1
        return AgentResult(
            output={"model_text": json.dumps(self._payload, ensure_ascii=True)},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"agent": "static-json-draft"},
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator:
        del prompt, request_id, dependencies
        raise NotImplementedError


def _write_dataset(*, filename: str) -> str:
    path = Path("artifacts/tests") / filename
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
                "rows": context["dependency_results"]["describe_dataset"]["output"]["result"][
                    "rows"
                ],
                "sample_count": context["dependency_results"]["load_sample"]["output"]["result"][
                    "count"
                ],
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
                "report_path": context["dependency_results"]["persist_report"]["output"]["result"][
                    "path"
                ]
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


def _mixed_branching_steps(*, agent_name: str) -> list[LogicStep | AgentStep | ToolStep]:
    return [
        LogicStep(
            step_id="router",
            handler=lambda context: {
                "route": (
                    "template_path"
                    if str(context["prompt"]).lower().startswith("template:")
                    else "agent_path"
                )
            },
            route_map={"agent_path": ("draft_agent",), "template_path": ("draft_template",)},
        ),
        AgentStep(
            step_id="draft_agent",
            agent_name=agent_name,
            dependencies=("router",),
            prompt_builder=lambda context: str(context["prompt"]),
        ),
        ToolStep(
            step_id="parse_agent_json",
            tool_name="text.extract_json",
            dependencies=("draft_agent",),
            input_builder=lambda context: {
                "text": context["dependency_results"]["draft_agent"]["output"]["output"][
                    "model_text"
                ]
            },
        ),
        LogicStep(
            step_id="finalize_agent",
            dependencies=("parse_agent_json",),
            handler=lambda context: {
                "branch": "agent",
                "title": context["dependency_results"]["parse_agent_json"]["output"]["result"][
                    "json"
                ].get("title", ""),
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


def test_pure_tool_workflow_accepts_user_defined_steps_with_inputs() -> None:
    dataset_path = _write_dataset(filename="pure_arbitrary_dataset.csv")
    workflow = pure_tool_workflow(
        tool_runtime=UnifiedToolRuntime(),
        steps=_pure_dataset_steps(),
        input_schema=_pure_input_schema(),
    )

    result = workflow.run(
        inputs={
            "dataset_csv_path": dataset_path,
            "quality_report_path": "artifacts/tests/pure_arbitrary_report.json",
            "required_columns": ["participant_id", "study_arm"],
            "sample_nrows": 3,
            "max_missing_ratio_per_column": 0.3,
        },
        request_id="test-pure-arbitrary",
    )

    assert result.success
    assert result.step_results["finalize"].status == "completed"
    assert str(result.step_results["finalize"].output["report_path"]).endswith(
        "artifacts/tests/pure_arbitrary_report.json"
    )


def test_pure_tool_workflow_validates_inputs_with_schema_hook() -> None:
    dataset_path = _write_dataset(filename="pure_schema_dataset.csv")
    workflow = pure_tool_workflow(
        tool_runtime=UnifiedToolRuntime(),
        steps=_pure_dataset_steps(),
        input_schema=_pure_input_schema(),
    )

    with pytest.raises(SchemaValidationError):
        workflow.run(
            inputs={
                "dataset_csv_path": dataset_path,
                "quality_report_path": "artifacts/tests/pure_schema_report.json",
                "required_columns": ["participant_id"],
                "sample_nrows": "3",
                "max_missing_ratio_per_column": 0.2,
            }
        )


def test_pure_tool_workflow_without_schema_allows_arbitrary_inputs() -> None:
    steps = [
        LogicStep(
            step_id="echo_inputs",
            handler=lambda context: {"inputs_snapshot": dict(context["inputs"])},
        )
    ]
    workflow = pure_tool_workflow(
        tool_runtime=UnifiedToolRuntime(),
        steps=steps,
    )

    result = workflow.run(inputs={"free_form": {"a": 1, "b": 2}}, request_id="test-pure-free")

    assert result.success
    assert result.step_results["echo_inputs"].output["inputs_snapshot"]["free_form"]["a"] == 1


def test_mixed_workflow_requires_non_empty_agents_and_steps() -> None:
    with pytest.raises(ValueError, match="agents"):
        mixed_agent_workflow(
            tool_runtime=UnifiedToolRuntime(),
            agents={},
            steps=[LogicStep(step_id="noop", handler=lambda context: {})],
        )

    with pytest.raises(ValueError, match="steps"):
        mixed_agent_workflow(
            tool_runtime=UnifiedToolRuntime(),
            agents={"any": _StaticJsonDraftAgent(payload={"title": "x"})},
            steps=[],
        )


def test_mixed_workflow_executes_user_defined_branching_steps() -> None:
    writer_agent = _StaticJsonDraftAgent(
        payload={"title": "Agent title", "summary": "Agent summary"}
    )
    workflow = mixed_agent_workflow(
        tool_runtime=UnifiedToolRuntime(),
        agents={"writer_agent": writer_agent},
        steps=_mixed_branching_steps(agent_name="writer_agent"),
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


def test_mixed_workflow_injects_prompt_and_preserves_base_context() -> None:
    writer_agent = _StaticJsonDraftAgent(payload={"title": "ignored"})
    custom_steps = [
        AgentStep(
            step_id="delegate",
            agent_name="analyst_agent",
            prompt_builder=lambda context: f"{context['base_tag']}::{context['prompt']}",
        ),
        LogicStep(
            step_id="finalize",
            dependencies=("delegate",),
            handler=lambda context: {
                "base_tag": context["base_tag"],
                "prompt_seen": context["prompt"],
                "model_text": context["dependency_results"]["delegate"]["output"]["output"][
                    "model_text"
                ],
            },
        ),
    ]
    workflow = mixed_agent_workflow(
        tool_runtime=UnifiedToolRuntime(),
        agents={"analyst_agent": writer_agent},
        steps=custom_steps,
        base_context={"base_tag": "custom"},
    )

    result = workflow.run(
        "Produce a short custom mixed-workflow brief.",
        request_id="test-mixed-context-injection",
    )

    assert result.success
    assert result.step_results["finalize"].output["base_tag"] == "custom"
    assert result.step_results["finalize"].output["prompt_seen"] == (
        "Produce a short custom mixed-workflow brief."
    )
