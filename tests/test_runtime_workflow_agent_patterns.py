"""Workflow pattern contract tests."""

from __future__ import annotations

import json

from design_research_agents.contracts.memory import MemoryWriteRecord
from design_research_agents.memory.stores.sqlite_store import SQLiteMemoryStore
from design_research_agents.tools import Toolbox
from design_research_agents.workflow import (
    NetworkedPattern,
    PlannerExecutorPattern,
    RagReasoningPattern,
    ReflexionPattern,
    RouterPattern,
)
from tests.helpers.workflow_stubs import SequenceLLMClient, StaticMarkerAgent


def test_plan_execute_workflow_runs_planner_then_executor() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "compute",
                            "instruction": "Compute 6 * 7.",
                            "success_criteria": "Return numeric result.",
                        }
                    ]
                }
            ),
            "\n".join(
                [
                    'calc = call_tool("calculator", {"expression": "6 * 7"})',
                    'final_output = {"result": calc["result"]}',
                ]
            ),
        ]
    )
    workflow = PlannerExecutorPattern(
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_iterations=2,
    )

    result = workflow.run("Compute 6 * 7.")

    assert result.success
    assert result.output["steps_executed"] == 1
    assert result.output["final_output"]["result"] == 42.0
    assert result.metadata["runtime"]["resolved_mode"] == "plan_execute"


def test_propose_and_critique_workflow_stops_on_approval() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            "Draft v1",
            json.dumps(
                {
                    "approved": False,
                    "feedback": "Add more detail.",
                    "revision_goals": ["expand rationale"],
                }
            ),
            "Draft v2 with more detail",
            json.dumps(
                {
                    "approved": True,
                    "feedback": "Looks good.",
                    "revision_goals": [],
                }
            ),
        ]
    )
    workflow = ReflexionPattern(
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_iterations=3,
    )

    result = workflow.run("Write a short design summary.")

    assert result.success
    assert result.output["approved"] is True
    assert result.output["terminated_reason"] == "approved"
    assert len(result.output["critique_iterations"]) == 2


def test_agent_routing_workflow_selects_and_executes_named_alternative() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            '{"tool_names": ["alt_two"], "reason": "best fit"}',
        ]
    )
    workflow = RouterPattern(
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        alternatives={
            "alt_one": StaticMarkerAgent(marker="one"),
            "alt_two": StaticMarkerAgent(marker="two"),
        },
    )

    result = workflow.run("Handle this request.")

    assert result.success
    assert result.output["agent_marker"] == "two"
    assert result.output["agent_routing_selected_alternative"] == "alt_two"
    assert result.metadata["agent_routing"]["selected_alternative"] == "alt_two"


def test_networked_pattern_runs_peer_only_round_and_reports_order() -> None:
    workflow = NetworkedPattern(
        peers={
            "peer_b": StaticMarkerAgent(marker="b"),
            "peer_a": StaticMarkerAgent(marker="a"),
        },
        max_rounds=1,
    )

    result = workflow.run("Coordinate this design task.")

    assert result.success
    assert result.output["terminated_reason"] == "max_rounds_reached"
    assert result.output["rounds_executed"] == 1
    assert result.metadata["peer_order"] == ["peer_a", "peer_b"]


def test_rag_reasoning_pattern_reads_memory_and_runs_delegate(tmp_path) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write(
        [MemoryWriteRecord(content="Context: prioritize safety.")],
        namespace="design",
    )
    workflow = RagReasoningPattern(
        reasoning_delegate=StaticMarkerAgent(marker="reasoned"),
        memory_store=store,
        memory_namespace="design",
        memory_top_k=2,
        write_back=False,
    )

    result = workflow.run("Draft a safe design plan.")
    store.close()

    assert result.success
    assert result.output["retrieval"]["count"] >= 1
    assert result.output["reasoning"]["output"]["agent_marker"] == "reasoned"
