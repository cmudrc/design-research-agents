from __future__ import annotations

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
    TERMINATED_STEP_FAILURE,
)
from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.agent import MultiStepAgent


class _SequenceLLMClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.chat_calls = 0

    def default_model(self) -> str:
        return "test-model"

    def close(self) -> None:
        return None

    def __enter__(self) -> _SequenceLLMClient:
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
            request.messages,
            model=request.model or self.default_model(),
            params=LLMChatParams(provider_options=dict(request.provider_options)),
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
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del dependencies
        payload = dict(input)
        self.calls.append((tool_name, payload, request_id))
        if tool_name == "sum":
            return ToolResult(
                tool_name=tool_name,
                ok=True,
                result={"value": int(payload.get("a", 0)) + int(payload.get("b", 0))},
            )
        return ToolResult(tool_name=tool_name, ok=False, result={}, error="boom")

    def close(self) -> None:
        return None

    def __enter__(self) -> _StubToolRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


def test_multi_step_code_tool_call_then_final_answer() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            "\n".join(
                [
                    'result = call_tool("sum", {"a": 2, "b": 3})',
                    'final_output = {"value": result["value"]}',
                ]
            ),
            'final_answer({"value": 5})',
        ]
    )
    tool_runtime = _StubToolRuntime()
    agent = MultiStepAgent(
        mode="code",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
    )

    result = agent.run("Compute 2+3")

    assert result.success is True
    assert result.output["steps_executed"] == 2
    assert result.output["terminated_reason"] == TERMINATED_COMPLETED
    assert result.output["final_output"] == {"value": 5}
    assert result.output["step_outputs"][0]["action_type"] == "tool_call"
    assert result.output["step_outputs"][1]["action_type"] == "final_answer"
    assert len(tool_runtime.calls) == 1
    assert llm_client.chat_calls == 2


def test_multi_step_code_tool_call_and_final_answer_in_same_step() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            "\n".join(
                [
                    'result = call_tool("sum", {"a": 4, "b": 5})',
                    'final_answer({"value": result["value"]})',
                ]
            )
        ]
    )
    tool_runtime = _StubToolRuntime()
    agent = MultiStepAgent(
        mode="code",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=2,
    )

    result = agent.run("Compute 4+5")

    assert result.success is True
    assert result.output["steps_executed"] == 1
    assert result.output["terminated_reason"] == TERMINATED_COMPLETED
    assert result.output["final_output"] == {"value": 9}
    assert len(tool_runtime.calls) == 1


def test_multi_step_code_max_steps_without_final_answer_is_incomplete() -> None:
    llm_client = _SequenceLLMClient(
        responses=[
            "\n".join(
                [
                    'result = call_tool("sum", {"a": 2, "b": 3})',
                    'final_output = {"value": result["value"]}',
                ]
            )
        ]
    )
    tool_runtime = _StubToolRuntime()
    agent = MultiStepAgent(
        mode="code",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=1,
    )

    result = agent.run("Compute 2+3")

    assert result.success is False
    assert result.output["steps_executed"] == 1
    assert result.output["terminated_reason"] == TERMINATED_MAX_STEPS_REACHED
    assert result.output["final_output"] == {}
    assert len(tool_runtime.calls) == 1


def test_multi_step_code_step_failure_still_respects_stop_on_step_failure() -> None:
    llm_client = _SequenceLLMClient(responses=['call_tool("fail", {})'])
    agent = MultiStepAgent(
        mode="code",
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
