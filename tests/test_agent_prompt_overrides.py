from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from design_research_agents._contracts._llm import (
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.agent import MultiStepAgent
from design_research_agents.patterns import PlannerExecutorPattern
from design_research_agents.tools import Toolbox


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


class _ArglessToolRuntime(ToolRuntime):
    def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="noop",
                description="No-op",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={"type": "object", "additionalProperties": True},
            )
        ]

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del input_dict, request_id, dependencies
        return ToolResult(tool_name=tool_name, ok=True, result={})


class _EmptyToolRuntime(ToolRuntime):
    def list_tools(self) -> list[ToolSpec]:
        return []

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del tool_name, input_dict, request_id, dependencies
        raise AssertionError("invoke should not be called for empty runtime")


def test_multi_step_agent_rejects_missing_runtime_for_json_or_code_modes() -> None:
    llm_client = _SequenceLLMClient(response_texts=['{"continue": false, "thought": "done"}'])

    with pytest.raises(ValueError, match="tool_runtime is required"):
        MultiStepAgent(mode="json", llm_client=llm_client)

    with pytest.raises(ValueError, match="tool_runtime is required"):
        MultiStepAgent(mode="code", llm_client=llm_client)


def test_multi_step_agent_rejects_empty_runtime_for_json_or_code_modes() -> None:
    llm_client = _SequenceLLMClient(response_texts=['{"continue": false, "thought": "done"}'])
    empty_runtime = _EmptyToolRuntime()

    with pytest.raises(ValueError, match="must expose at least one tool"):
        MultiStepAgent(mode="json", llm_client=llm_client, tool_runtime=empty_runtime)

    with pytest.raises(ValueError, match="must expose at least one tool"):
        MultiStepAgent(mode="code", llm_client=llm_client, tool_runtime=empty_runtime)


def test_multi_step_json_tool_agent_rejects_invalid_alternatives_prompt_target() -> None:
    with pytest.raises(ValueError, match="alternatives_prompt_target"):
        MultiStepAgent(
            mode="json",
            llm_client=_SequenceLLMClient(response_texts=['{"tool_name":"calculator","tool_input":{}}']),
            tool_runtime=Toolbox(),
            alternatives_prompt_target="invalid",
        )


def test_multi_step_json_agent_rejects_unmatched_allowed_tools() -> None:
    agent = MultiStepAgent(
        mode="json",
        llm_client=_SequenceLLMClient(response_texts=['{"tool_name":"calculator","tool_input":{},"reason":"x"}']),
        tool_runtime=_ArglessToolRuntime(),
        allowed_tools=["unknown_tool"],
    )
    with pytest.raises(ValueError, match="allowed_tools"):
        agent.run("route request")


def test_multi_step_code_agent_rejects_empty_prompt_override() -> None:
    with pytest.raises(ValueError, match="continuation_system_prompt"):
        MultiStepAgent(
            mode="code",
            llm_client=_SequenceLLMClient(response_texts=["final_output = {}"]),
            tool_runtime=Toolbox(),
            continuation_system_prompt="   ",
        )


def test_plan_execute_workflow_template_override_supports_task_prompt_variable() -> None:
    workflow = PlannerExecutorPattern(
        llm_client=_SequenceLLMClient(
            response_texts=[
                '{"steps":[{"step_id":"one","instruction":"Count words in a short phrase.",'
                '"success_criteria":"Return word count"}]}',
                '{"continue": true, "thought": "run step"}',
                'stats = call_tool("text.word_count", {"text": "design research"})\n'
                'final_output = {"result": stats["word_count"]}',
            ]
        ),
        tool_runtime=Toolbox(),
        planner_user_prompt_template="Task block:\n$task_prompt",
    )
    result = workflow.run("Count words in design research.")
    assert result.success


def test_plan_execute_workflow_template_override_rejects_missing_variables() -> None:
    workflow = PlannerExecutorPattern(
        llm_client=_SequenceLLMClient(response_texts=['{"steps":[]}']),
        tool_runtime=Toolbox(),
        planner_user_prompt_template="Task block:\n$unknown_key",
    )
    with pytest.raises(ValueError, match="unknown_key"):
        workflow.run("Compute 6 * 7.")
