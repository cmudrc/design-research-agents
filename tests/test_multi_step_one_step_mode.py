from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents._contracts._llm import (
    LLMChatParams,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents._contracts._termination import (
    TERMINATED_COMPLETED,
    TERMINATED_MAX_STEPS_REACHED,
)
from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.agent import MultiStepAgent
from design_research_agents.tools import Toolbox


class _SequenceChatLLMClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.chat_calls = 0

    def default_model(self) -> str:
        return "test-model"

    def close(self) -> None:
        return None

    def __enter__(self) -> _SequenceChatLLMClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None

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
            list(request.messages),
            model=request.model or self.default_model(),
            params=LLMChatParams(),
        )


class _JsonToolRuntime(ToolRuntime):
    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="sum",
                description="Add numbers",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={"type": "object", "additionalProperties": True},
            ),
        )

    def invoke(
        self,
        tool_name: str,
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        return ToolResult(
            tool_name=tool_name,
            ok=True,
            result={"value": 5},
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> _JsonToolRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


def test_multi_step_json_one_step_mode_runs_single_action_step() -> None:
    llm_client = _SequenceChatLLMClient(
        responses=[
            json.dumps(
                {
                    "tool_name": "sum",
                    "tool_input": {"a": 2, "b": 3},
                }
            )
        ]
    )
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=_JsonToolRuntime(),
        max_steps=1,
    )

    result = agent.run("Compute 2+3")

    assert result.success is False
    assert result.output["steps_executed"] == 1
    assert result.output["terminated_reason"] == TERMINATED_MAX_STEPS_REACHED
    assert result.output["final_output"] == {}
    assert len(result.tool_results) == 1
    assert llm_client.chat_calls == 1


def test_multi_step_json_one_step_mode_can_finish_with_immediate_final_answer() -> None:
    llm_client = _SequenceChatLLMClient(
        responses=[
            json.dumps(
                {
                    "tool_name": "final_answer",
                    "tool_input": {"word_count": 2},
                }
            ),
        ]
    )
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=1,
    )

    result = agent.run("Count words in a short phrase.")

    assert result.success is True
    assert result.output["steps_executed"] == 1
    assert result.output["terminated_reason"] == TERMINATED_COMPLETED
    assert result.output["final_output"]["word_count"] == 2
    assert len(result.tool_results) == 0
    assert llm_client.chat_calls == 1


def test_multi_step_code_one_step_mode_runs_single_code_step() -> None:
    llm_client = _SequenceChatLLMClient(
        responses=[
            "\n".join(
                [
                    'stats = call_tool("text.word_count", {"text": "design research"})',
                    'final_answer({"result": stats["word_count"]})',
                ]
            ),
        ]
    )
    agent = MultiStepAgent(
        mode="code",
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=1,
    )

    result = agent.run("Count words in a short phrase.")

    assert result.success is True
    assert result.output["steps_executed"] == 1
    assert result.output["terminated_reason"] == TERMINATED_COMPLETED
    assert result.output["final_output"] == {"result": 2}
    assert len(result.tool_results) == 1
    assert llm_client.chat_calls == 1
