from __future__ import annotations

from collections.abc import Iterator

import pytest

from design_research_agents.agent import (
    SingleStepCodeToolCallingAgent,
    SingleStepJsonToolCallingAgent,
    SingleStepToolRouterAgent,
)
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents.tools import Toolbox
from design_research_agents.workflow import PlannerExecutorPattern


class _SequenceLLMClient:
    def __init__(self, *, response_texts: list[str]) -> None:
        self._responses = list(response_texts)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        if not self._responses:
            raise AssertionError("No more stubbed responses available.")
        return LLMResponse(
            model=model,
            text=self._responses.pop(0),
            provider="test-sequence",
            latency_ms=1,
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.chat(
            list(request.messages),
            model=request.model or self.default_model(),
            params=LLMChatParams(),
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        return "test-model"


def test_single_step_json_tool_agent_rejects_invalid_alternatives_prompt_target() -> None:
    with pytest.raises(ValueError, match="alternatives_prompt_target"):
        SingleStepJsonToolCallingAgent(
            llm_client=_SequenceLLMClient(
                response_texts=['{"tool_name":"calculator","tool_input":{}}']
            ),
            tool_runtime=Toolbox(),
            alternatives_prompt_target="invalid",
        )


def test_single_step_json_tool_agent_rejects_unmatched_allowed_tools() -> None:
    with pytest.raises(ValueError, match="allowed_tools"):
        SingleStepJsonToolCallingAgent(
            llm_client=_SequenceLLMClient(
                response_texts=['{"tool_name":"calculator","tool_input":{}}']
            ),
            tool_runtime=Toolbox(),
            allowed_tools=["does_not_exist"],
        )


def test_single_step_router_agent_rejects_unmatched_allowed_routes() -> None:
    with pytest.raises(ValueError, match="allowed_routes"):
        SingleStepToolRouterAgent(
            llm_client=_SequenceLLMClient(response_texts=['{"selection":0,"reason":"x"}']),
            tool_runtime=Toolbox(),
            allowed_routes=["unknown_route"],
        )


def test_single_step_code_agent_rejects_empty_prompt_override() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        SingleStepCodeToolCallingAgent(
            llm_client=_SequenceLLMClient(response_texts=["final_output = {}"]),
            tool_runtime=Toolbox(),
            system_prompt="   ",
        )


def test_plan_execute_workflow_template_override_supports_task_prompt_variable() -> None:
    workflow = PlannerExecutorPattern(
        llm_client=_SequenceLLMClient(
            response_texts=[
                '{"steps":[{"step_id":"one","instruction":"Compute 6 * 7.",'
                '"success_criteria":"Return result"}]}',
                'calc = call_tool("calculator", {"expression": "6 * 7"})\n'
                'final_output = {"result": calc["result"]}',
            ]
        ),
        tool_runtime=Toolbox(),
        plan_execute_planner_user_prompt_template="Task block:\n$task_prompt",
    )
    result = workflow.run("Compute 6 * 7.")
    assert result.success


def test_plan_execute_workflow_template_override_rejects_missing_variables() -> None:
    workflow = PlannerExecutorPattern(
        llm_client=_SequenceLLMClient(response_texts=['{"steps":[]}']),
        tool_runtime=Toolbox(),
        plan_execute_planner_user_prompt_template="Task block:\n$unknown_key",
    )
    with pytest.raises(ValueError, match="unknown_key"):
        workflow.run("Compute 6 * 7.")
