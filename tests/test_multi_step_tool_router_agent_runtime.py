from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents.agent.implementations.multi_step_tool_router_agent import (
    MultiStepToolRouterAgent,
)
from design_research_agents.agent.internal.multi_step_tool_router_helpers import (
    ToolRouterStepDecision,
)
from design_research_agents.agent.internal.router_agent_helpers import ToolAlternative
from design_research_agents.contracts.llm import LLMChatParams, LLMMessage, LLMResponse
from design_research_agents.contracts.termination import (
    TERMINATED_INVALID_ROUTE_SELECTION,
    TERMINATED_INVALID_STEP_OUTPUT,
    TERMINATED_STEP_FAILURE,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec


class _SequenceLLMClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.chat_calls = 0

    def default_model(self) -> str:
        return "test-model"

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        if not self._responses:
            raise AssertionError("No stubbed model responses remain.")
        self.chat_calls += 1
        return LLMResponse(model=model, text=self._responses.pop(0), provider="test")


class _StubToolRuntime(ToolRuntime):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], str]] = []

    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="sum",
                description="Add numbers",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "additionalProperties": True},
            ),
            ToolSpec(
                name="fail",
                description="Always fails",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "additionalProperties": True},
            ),
        )

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del dependencies
        payload = dict(input_dict)
        self.calls.append((tool_name, payload, request_id))
        if tool_name == "sum":
            return ToolResult(
                tool_name=tool_name,
                ok=True,
                result={"value": int(payload.get("a", 0)) + int(payload.get("b", 0))},
            )
        return ToolResult(tool_name=tool_name, ok=False, result={}, error="boom")


def test_multi_step_tool_router_tool_call_then_stop() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            json.dumps(
                {
                    "action": "TOOL_CALL",
                    "tool_names": ["sum"],
                    "tool_input": {"a": 2, "b": 3},
                    "reason": "compute",
                }
            ),
            json.dumps(
                {
                    "action": "STOP",
                    "final_output": {"answer": 5},
                    "reason": "done",
                }
            ),
        ]
    )
    tool_runtime = _StubToolRuntime()
    agent = MultiStepToolRouterAgent(llm_client=llm_client, tool_runtime=tool_runtime, max_steps=3)

    result = agent.run("Compute 2+3", request_id="req-router-success")

    assert result.success is True
    assert result.output["steps_executed"] == 2
    assert result.output["final_output"] == {"answer": 5}
    assert result.output["terminated_reason"] == "stop:model"
    assert len(result.tool_results) == 1
    assert len(tool_runtime.calls) == 1
    assert tool_runtime.calls[0][0] == "sum"
    assert llm_client.chat_calls == 2


def test_multi_step_tool_router_invalid_step_output_fails() -> None:
    llm_client = _SequenceLLMClient(responses=[json.dumps({"continue": True, "thought": "legacy"})])
    agent = MultiStepToolRouterAgent(
        llm_client=llm_client, tool_runtime=_StubToolRuntime(), max_steps=2
    )

    result = agent.run("Bad payload")

    assert result.success is False
    assert result.output["terminated_reason"] == TERMINATED_INVALID_STEP_OUTPUT
    assert result.metadata["stage"] == "step_decision"


def test_multi_step_tool_router_invalid_route_selection_fails() -> None:
    llm_client = _SequenceLLMClient(
        responses=[json.dumps({"action": "TOOL_CALL", "tool_names": ["missing"], "reason": "x"})]
    )
    tool_runtime = _StubToolRuntime()
    agent = MultiStepToolRouterAgent(llm_client=llm_client, tool_runtime=tool_runtime, max_steps=2)

    result = agent.run("Route to unknown tool")

    assert result.success is False
    assert result.output["terminated_reason"] == TERMINATED_INVALID_ROUTE_SELECTION
    assert tool_runtime.calls == []


def test_multi_step_tool_router_step_failure_stops_when_configured() -> None:
    llm_client = _SequenceLLMClient(
        responses=[json.dumps({"action": "TOOL_CALL", "tool_names": ["fail"], "reason": "x"})]
    )
    agent = MultiStepToolRouterAgent(
        llm_client=llm_client,
        tool_runtime=_StubToolRuntime(),
        max_steps=2,
        stop_on_step_failure=True,
    )

    result = agent.run("Fail fast")

    assert result.success is False
    assert result.output["steps_executed"] == 1
    assert result.output["terminated_reason"] == TERMINATED_STEP_FAILURE
    assert result.metadata["stage"] == "step_execution"


def test_multi_step_tool_router_internal_step_failure_can_continue_when_disabled() -> None:
    tool_runtime = _StubToolRuntime()
    agent = MultiStepToolRouterAgent(
        llm_client=_SequenceLLMClient(responses=[]),
        tool_runtime=tool_runtime,
        max_steps=2,
        stop_on_step_failure=False,
    )
    state = agent._run_tool_call_step(
        step_number=1,
        parsed_step=ToolRouterStepDecision(
            action="TOOL_CALL",
            tool_names=("fail",),
            tool_input={"x": 1},
            final_output=None,
            reason="exercise branch",
            source="model",
        ),
        alternatives=[
            ToolAlternative(
                tool_name="fail",
                description="Always fails",
                input_schema={"type": "object"},
            )
        ],
        normalized_input={"prompt": "task prompt"},
        request_id="req-router-branch",
        dependencies={},
        memory=[],
        step_outputs=[],
        tool_results=[],
        final_output={},
        last_model_response=None,
        stop_on_step_failure=False,
    )

    assert state["terminated_reason"] == TERMINATED_STEP_FAILURE
    assert state["should_continue"] is True
    assert state["fatal_error"] is None
