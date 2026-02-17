"""AgentRuntime and workflow pattern contract tests."""

from __future__ import annotations

import json

import pytest

from design_research_agents.agent import (
    AgentRuntime,
    MultiStepCodeToolCallingAgent,
    MultiStepJsonToolCallingAgent,
    RuntimeControls,
)
from design_research_agents.tools import Toolbox
from design_research_agents.workflow import PlannerExecutorPattern, ReflexionPattern, RouterPattern
from tests.helpers.workflow_stubs import SequenceLLMClient, StaticMarkerAgent


def test_agent_runtime_react_mode_aliases_multi_step_agent() -> None:
    response_texts = [
        '{"continue": true, "thought": "start"}',
        "\n".join(
            [
                'calc = call_tool("calculator", {"expression": "6 * 7"})',
                'final_output = {"result": calc["result"]}',
            ]
        ),
        '{"continue": false, "thought": "done"}',
    ]

    llm_runtime = SequenceLLMClient(response_texts=response_texts)
    llm_direct = SequenceLLMClient(response_texts=list(response_texts))
    tool_runtime = Toolbox()

    runtime_agent = AgentRuntime(
        llm_client=llm_runtime,
        tool_runtime=tool_runtime,
        mode="react",
        controls=RuntimeControls(max_steps=3),
    )
    direct_agent = MultiStepCodeToolCallingAgent(
        llm_client=llm_direct,
        tool_runtime=tool_runtime,
        max_steps=3,
    )

    runtime_result = runtime_agent.run("Compute 6 * 7.")
    direct_result = direct_agent.run("Compute 6 * 7.")

    assert runtime_result.success == direct_result.success
    assert runtime_result.output["final_output"] == direct_result.output["final_output"]
    assert runtime_result.output["terminated_reason"] == direct_result.output["terminated_reason"]
    assert (
        runtime_result.metadata["runtime"]["resolved_mode"] == "multi_step_code_tool_calling_agent"
    )


def test_multi_step_json_tool_calling_agent_runs_one_successful_step() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            '{"continue": true, "thought": "start"}',
            '{"tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}',
            '{"continue": false, "thought": "done"}',
        ]
    )
    agent = MultiStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
    )

    result = agent.run("Compute 6 * 7.")

    assert result.success
    assert result.output["steps_executed"] == 1
    assert result.output["final_output"]["result"] == 42.0
    assert result.output["terminated_reason"] == "continuation_stopped:model"


def test_multi_step_json_tool_calling_agent_uses_fallback_continuation_on_invalid_json() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            "invalid continuation payload",
            '{"tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}',
            "invalid continuation payload",
        ]
    )
    agent = MultiStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
    )

    result = agent.run("Compute 6 * 7.")

    assert result.success
    assert result.output["steps_executed"] == 1
    assert result.output["terminated_reason"] == "continuation_stopped:fallback"
    continuation = result.metadata["continuation"]
    assert isinstance(continuation, list)
    assert continuation[0]["source"] == "fallback"


def test_multi_step_json_tool_calling_agent_stops_on_step_failure() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            '{"continue": true, "thought": "start"}',
            '{"tool_name": "calculator", "tool_input": {"expression": "1 / 0"}}',
        ]
    )
    agent = MultiStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
        stop_on_step_failure=True,
    )

    result = agent.run("Compute 6 * 7.")

    assert result.success is False
    assert result.output["terminated_reason"] == "step_failure"


def test_agent_runtime_rejects_non_react_mode_with_migration_message() -> None:
    with pytest.raises(ValueError, match="mode='react' only"):
        AgentRuntime(
            llm_client=SequenceLLMClient(response_texts=[]),
            tool_runtime=Toolbox(),
            mode="plan_execute",
        )


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
        controls=RuntimeControls(max_iterations=2),
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
        controls=RuntimeControls(max_iterations=3),
    )

    result = workflow.run("Write a short design summary.")

    assert result.success
    assert result.output["approved"] is True
    assert result.output["terminated_reason"] == "approved"
    assert len(result.output["critique_iterations"]) == 2


def test_agent_routing_workflow_selects_and_executes_named_alternative() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            '{"selection": "alt_two", "reason": "best fit"}',
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


def test_plan_execute_workflow_stream_emits_delta_then_completed() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "s1",
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
    )

    events = list(workflow.run_stream("Compute 6 * 7."))

    assert [event.kind for event in events] == ["delta", "completed"]
    assert events[1].result is not None
    assert events[1].result.success
