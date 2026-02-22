from __future__ import annotations

import json
from collections.abc import Sequence

from design_research_agents._contracts._llm import LLMChatParams, LLMMessage, LLMResponse
from design_research_agents._contracts._termination import (
    TERMINATED_CONTROLLER_INVALID_PAYLOAD,
    TERMINATED_MAX_STEPS_REACHED,
)
from design_research_agents._implementations._agents._multi_step_agent import (
    _coerce_state_records,
    _parse_controller_decision,
)
from design_research_agents.agent import MultiStepAgent


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


def test_multi_step_direct_llm_agent_continue_then_stop() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            json.dumps({"decision": "CONTINUE", "content": "draft answer"}),
            json.dumps(
                {
                    "decision": "STOP",
                    "content": "final answer",
                    "final_output": "42",
                    "reason": "done",
                }
            ),
        ]
    )
    agent = MultiStepAgent(mode="direct", llm_client=llm_client, max_steps=3)

    result = agent.run(
        "What is 6*7?",
        request_id="req-direct-success",
        dependencies={"dependency": "ok"},
    )

    assert result.success is True
    assert agent.workflow is not None
    assert result.output["final_output"] == "42"
    assert isinstance(result.output["workflow"], dict)
    assert isinstance(result.output["artifacts"], list)
    assert result.output["steps_executed"] == 2
    assert result.output["terminated_reason"] == "stop:model"
    assert result.output["step_outputs"][0]["decision"] == "CONTINUE"
    assert result.output["step_outputs"][1]["decision"] == "STOP"
    assert result.metadata["request_id"] == "req-direct-success"
    assert result.metadata["dependency_keys"] == ["dependency"]
    assert llm_client.chat_calls == 2


def test_multi_step_direct_llm_agent_invalid_controller_payload_fails() -> None:
    llm_client = _SequenceLLMClient(responses=[json.dumps({"continue": True, "thought": "legacy"})])
    agent = MultiStepAgent(mode="direct", llm_client=llm_client, max_steps=2)

    result = agent.run("Test invalid payload")

    assert result.success is False
    assert result.output["terminated_reason"] == TERMINATED_CONTROLLER_INVALID_PAYLOAD
    assert result.metadata["stage"] == "controller_step"
    assert result.metadata["terminated_reason"] == TERMINATED_CONTROLLER_INVALID_PAYLOAD
    assert result.model_response is not None


def test_multi_step_direct_llm_agent_max_steps_reached_on_all_continue() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            json.dumps({"decision": "CONTINUE", "content": "first"}),
            json.dumps({"decision": "CONTINUE", "content": "second"}),
        ]
    )
    agent = MultiStepAgent(mode="direct", llm_client=llm_client, max_steps=2)

    result = agent.run("Continue until max steps")

    assert result.success is True
    assert result.output["steps_executed"] == 2
    assert result.output["final_output"] == "second"
    assert result.output["terminated_reason"] == TERMINATED_MAX_STEPS_REACHED


def test_multi_step_direct_llm_agent_parser_helpers_are_strict() -> None:
    parsed = _parse_controller_decision(json.dumps({"decision": "stop", "content": "done", "final_output": "ok"}))
    assert parsed is not None
    assert parsed.decision == "STOP"
    assert parsed.final_output == "ok"

    assert _parse_controller_decision(json.dumps({"continue": True, "thought": "legacy"})) is None
    assert _parse_controller_decision(json.dumps({"decision": "WAIT", "content": "bad"})) is None
    assert _parse_controller_decision("{not-json") is None

    assert _coerce_state_records("not-a-list") == []
    assert _coerce_state_records([{"ok": True}, "bad", 3]) == [{"ok": True}]
