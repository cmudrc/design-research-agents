"""Runtime checks for workflow-native single-step agents."""

from __future__ import annotations

from design_research_agents.agent import (
    SingleStepCodeToolCallingAgent,
    SingleStepDirectLLMAgent,
    SingleStepJsonToolCallingAgent,
    SingleStepToolRouterAgent,
)
from design_research_agents.tools import Toolbox
from tests.helpers.workflow_stubs import SequenceLLMClient


def test_single_step_direct_agent_emits_workflow_first_envelope() -> None:
    agent = SingleStepDirectLLMAgent(
        llm_client=SequenceLLMClient(response_texts=["Direct answer"]),
    )

    result = agent.run("Say something")

    assert result.success
    assert agent.workflow is not None
    assert isinstance(result.output["workflow"], dict)
    assert result.output["final_output"] == result.output["model_text"]
    assert isinstance(result.output["artifacts"], list)


def test_single_step_json_agent_emits_workflow_first_envelope() -> None:
    agent = SingleStepJsonToolCallingAgent(
        llm_client=SequenceLLMClient(
            response_texts=['{"tool_name":"calculator","tool_input":{"expression":"6 * 7"}}']
        ),
        tool_runtime=Toolbox(),
    )

    result = agent.run("Compute 6 * 7")

    assert result.success
    assert agent.workflow is not None
    assert isinstance(result.output["workflow"], dict)
    assert result.output["final_output"] == result.output["tool_output"]
    assert isinstance(result.output["artifacts"], list)


def test_single_step_router_agent_emits_workflow_first_envelope() -> None:
    agent = SingleStepToolRouterAgent(
        llm_client=SequenceLLMClient(
            response_texts=['{"tool_names":["calculator"],"reason":"best"}']
        ),
        tool_runtime=Toolbox(),
    )

    result = agent.run("Compute 5 + 5")

    assert result.success
    assert agent.workflow is not None
    assert isinstance(result.output["workflow"], dict)
    assert result.output["final_output"] == result.output["tool_output"]
    assert isinstance(result.output["artifacts"], list)


def test_single_step_code_agent_emits_workflow_first_envelope() -> None:
    agent = SingleStepCodeToolCallingAgent(
        llm_client=SequenceLLMClient(
            response_texts=[
                'calc = call_tool("calculator", {"expression": "3 * 14"})\n'
                'final_output = {"result": calc["result"]}'
            ]
        ),
        tool_runtime=Toolbox(),
    )

    result = agent.run("Compute 3 * 14")

    assert result.success
    assert agent.workflow is not None
    assert isinstance(result.output["workflow"], dict)
    assert isinstance(result.output["final_output"], dict)
    assert isinstance(result.output["tool_output"], dict)
    assert result.output["final_output"] == {"result": 42.0}
    assert result.output["tool_output"].get("result") == 42.0
    assert "generated_code" in result.output
    assert isinstance(result.output["artifacts"], list)
