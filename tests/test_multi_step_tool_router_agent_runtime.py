from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents.agent import MultiStepAgent
from design_research_agents.contracts.llm import LLMChatParams, LLMMessage, LLMRequest, LLMResponse
from design_research_agents.contracts.termination import TERMINATED_STEP_FAILURE
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

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.chat(
            request.messages,
            model=request.model or self.default_model(),
            params=LLMChatParams(
                response_schema=request.response_schema,
                provider_options=dict(request.provider_options),
            ),
        )


class _StubToolRuntime(ToolRuntime):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], str]] = []

    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="sum",
                description="Add numbers",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={"type": "object", "additionalProperties": True},
            ),
            ToolSpec(
                name="fail",
                description="Always fails",
                input_schema={"type": "object", "additionalProperties": False},
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


def test_multi_step_json_tool_call_then_continuation_stop() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            json.dumps({"continue": True, "thought": "compute"}),
            json.dumps({"tool_name": "sum", "tool_input": {"a": 2, "b": 3}}),
            json.dumps({"continue": False, "thought": "done"}),
        ]
    )
    tool_runtime = _StubToolRuntime()
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
    )

    result = agent.run("Compute 2+3", request_id="req-json-success")

    assert result.success is True
    assert agent.workflow is not None
    assert result.output["steps_executed"] == 1
    assert result.output["final_output"] == {"value": 5}
    assert isinstance(result.output["workflow"], dict)
    assert isinstance(result.output["artifacts"], list)
    assert str(result.output["terminated_reason"]).startswith("continuation_stopped")
    assert len(result.tool_results) == 1
    assert len(tool_runtime.calls) == 1
    assert tool_runtime.calls[0][0] == "sum"
    assert llm_client.chat_calls == 3


def test_multi_step_json_invalid_step_output_fails() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            json.dumps({"continue": True, "thought": "run"}),
            json.dumps({"unexpected": True}),
        ]
    )
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=_StubToolRuntime(),
        max_steps=2,
    )

    result = agent.run("Bad payload")

    assert result.success is False
    assert result.output["terminated_reason"] == TERMINATED_STEP_FAILURE


def test_multi_step_json_invalid_tool_selection_fails_without_invocation() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            json.dumps({"continue": True, "thought": "route"}),
            json.dumps({"tool_name": "missing", "tool_input": {}, "reason": "x"}),
        ]
    )
    tool_runtime = _StubToolRuntime()
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=2,
    )

    result = agent.run("Route to unknown tool")

    assert result.success is False
    assert result.output["terminated_reason"] == TERMINATED_STEP_FAILURE
    assert tool_runtime.calls == []


def test_multi_step_json_step_failure_stops_when_configured() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            json.dumps({"continue": True, "thought": "run"}),
            json.dumps({"tool_name": "fail", "tool_input": {"x": 1}}),
        ]
    )
    agent = MultiStepAgent(
        mode="json",
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


def test_multi_step_json_step_failure_can_continue_when_disabled() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            json.dumps({"continue": True, "thought": "first"}),
            json.dumps({"tool_name": "fail", "tool_input": {"x": 1}}),
            json.dumps({"continue": True, "thought": "second"}),
            json.dumps({"tool_name": "sum", "tool_input": {"a": 1, "b": 4}}),
        ]
    )
    tool_runtime = _StubToolRuntime()
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=2,
        stop_on_step_failure=False,
    )

    result = agent.run("Continue after failure")

    assert result.success is False
    assert result.output["steps_executed"] == 2
    assert len(tool_runtime.calls) == 2
    assert result.output["terminated_reason"] == "max_steps_reached"
