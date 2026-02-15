from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest

from design_research_agents.agent import (
    AgentRuntime,
    MultiStepCodeToolCallingAgent,
    MultiStepJsonToolCallingAgent,
    SingleStepCodeToolCallingAgent,
    SingleStepDirectLLMAgent,
    SingleStepJsonToolCallingAgent,
    SingleStepRouterAgent,
)
from design_research_agents.contracts.llm import LLMChatParams, LLMDelta, LLMMessage, LLMRequest
from design_research_agents.tools import Toolbox


class _EmptyDefaultModelClient:
    def default_model(self) -> str:
        return "   "

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ):
        del messages, model, params
        raise AssertionError("chat() should not be called when model resolution fails")

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ):
        del messages, model, params
        raise AssertionError("stream_chat() should not be called when model resolution fails")

    def generate(self, request: LLMRequest):
        del request
        raise AssertionError("generate() should not be called when model resolution fails")

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        del request
        raise AssertionError("stream() should not be called when model resolution fails")
        yield LLMDelta(text_delta="")


def test_agent_constructor_signatures_do_not_accept_model_kwarg() -> None:
    classes = (
        SingleStepDirectLLMAgent,
        SingleStepRouterAgent,
        SingleStepJsonToolCallingAgent,
        SingleStepCodeToolCallingAgent,
        MultiStepJsonToolCallingAgent,
        MultiStepCodeToolCallingAgent,
        AgentRuntime,
    )
    for cls in classes:
        assert "model" not in inspect.signature(cls.__init__).parameters


def test_direct_llm_agent_fails_when_llm_default_model_is_empty() -> None:
    with pytest.raises(ValueError, match=r"default_model\(\) returned an empty model id"):
        SingleStepDirectLLMAgent(llm_client=_EmptyDefaultModelClient()).run("Hello")


def test_agent_runtime_fails_when_llm_default_model_is_empty() -> None:
    with pytest.raises(ValueError, match=r"default_model\(\) returned an empty model id"):
        AgentRuntime(
            llm_client=_EmptyDefaultModelClient(),
            tool_runtime=Toolbox(),
            mode="plan_execute",
        ).run("Compute 1 + 1")
