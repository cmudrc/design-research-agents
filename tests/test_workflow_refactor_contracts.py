"""Contract checks for workflow-first refactor behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from design_research_agents._contracts._llm import LLMChatParams, LLMMessage, LLMResponse
from design_research_agents._contracts._workflow import (
    AgentStep,
    LogicStep,
    WorkflowArtifact,
    WorkflowArtifactSource,
)
from design_research_agents._implementations._patterns._tree_search import TreeSearchPattern
from design_research_agents.workflow import Workflow


class _StubLLMClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)

    def default_model(self) -> str:
        return "test-model"

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, model, params
        if not self._responses:
            raise AssertionError("No stubbed model responses remain.")
        return LLMResponse(model="test-model", text=self._responses.pop(0), provider="test")


def test_agent_step_rejects_legacy_name_based_delegate() -> None:
    with pytest.raises(TypeError):
        AgentStep(step_id="delegate", agent_name="legacy", prompt="Do work.")  # type: ignore[call-arg]


def test_workflow_rejects_agents_registry_constructor_kwarg() -> None:
    with pytest.raises(TypeError):
        Workflow(
            tool_runtime=None,
            input_schema={},
            steps=[LogicStep(step_id="ok", handler=lambda _context: {"ok": True})],
            agents={},  # type: ignore[call-arg]
        )


def test_workflow_accepts_pattern_delegate_on_agent_step() -> None:
    pattern = TreeSearchPattern(
        generator_delegate=lambda context: [{"candidate": context.get("task", "")}],
        evaluator_delegate=lambda context: 1.0,
        max_depth=1,
        branch_factor=1,
        beam_width=1,
    )
    workflow = Workflow(
        tool_runtime=None,
        steps=[
            AgentStep(
                step_id="delegate_pattern",
                delegate=pattern,
                prompt_builder=lambda context: str(context["prompt"]),
            )
        ],
    )

    result = workflow.run("route through pattern")

    assert result.success
    nested = result.step_results["delegate_pattern"].output
    assert nested["success"] is True
    assert "workflow" in nested["output"]


def test_workflow_artifact_contract_serializes_and_tracks_provenance() -> None:
    artifact = WorkflowArtifact(
        path="artifacts/tests/manual.json",
        mime="application/json",
        producer_step_id="manual_step",
        sources=(WorkflowArtifactSource(step_id="manual_step", field="manual"),),
    )
    artifact_dict = artifact.asdict()
    assert artifact_dict["producer_step_id"] == "manual_step"
    assert artifact_dict["sources"][0]["field"] == "manual"

    workflow = Workflow(
        tool_runtime=None,
        input_schema={},
        steps=[
            LogicStep(
                step_id="produce_artifacts",
                handler=lambda _context: {
                    "artifacts": [
                        {
                            "path": "artifacts/tests/from_output.json",
                            "mime": "application/json",
                            "title": "Output artifact",
                        }
                    ]
                },
                artifacts_builder=lambda _context: [
                    WorkflowArtifact(
                        path="artifacts/tests/from_builder.txt",
                        mime="text/plain",
                        producer_step_id="produce_artifacts",
                        sources=(
                            WorkflowArtifactSource(
                                step_id="produce_artifacts",
                                field="builder",
                            ),
                        ),
                    )
                ],
            )
        ],
    )

    result = workflow.run({})

    assert result.success
    artifacts = result.output["artifacts"]
    assert len(artifacts) == 2
    assert artifacts[0]["producer_step_id"] == "produce_artifacts"
    assert artifacts[0]["sources"][0]["step_id"] == "produce_artifacts"
    assert artifacts[1]["sources"][0]["field"] == "builder"


def test_pattern_outputs_include_workflow_first_envelope_fields() -> None:
    pattern = TreeSearchPattern(
        generator_delegate=lambda _context: [{"name": "alpha", "score_hint": 0.5}],
        evaluator_delegate=lambda context: float(dict(context.get("candidate", {})).get("score_hint", 0.0)),
        max_depth=1,
        branch_factor=1,
        beam_width=1,
    )
    result = pattern.run("envelope check")

    assert result.success
    assert isinstance(result.output["workflow"], dict)
    assert isinstance(result.output["final_output"], Mapping)
    assert isinstance(result.output["artifacts"], list)
