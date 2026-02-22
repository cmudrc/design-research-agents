from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest

from design_research_agents._contracts._llm import (
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
)
from design_research_agents.agent import (
    DirectLLMCall,
    MultiStepAgent,
)
from design_research_agents.patterns import (
    ConversationPattern,
    PlannerExecutorPattern,
    ReflexionPattern,
)
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
        DirectLLMCall,
        MultiStepAgent,
        ConversationPattern,
        PlannerExecutorPattern,
        ReflexionPattern,
    )
    for cls in classes:
        assert "model" not in inspect.signature(cls.__init__).parameters


def test_agent_constructor_signatures_expose_supported_kwargs() -> None:
    multi_code_params = inspect.signature(MultiStepAgent.__init__).parameters
    assert "execution_timeout_seconds" in multi_code_params

    planner_params = inspect.signature(PlannerExecutorPattern.__init__).parameters
    assert "max_iterations" in planner_params
    assert "max_tool_calls_per_step" in planner_params

    reflexion_params = inspect.signature(ReflexionPattern.__init__).parameters
    assert "max_iterations" in reflexion_params


def test_direct_llm_agent_fails_when_llm_default_model_is_empty() -> None:
    with pytest.raises(ValueError, match=r"default_model\(\) returned an empty model id"):
        DirectLLMCall(llm_client=_EmptyDefaultModelClient()).run("Hello")


def test_workflow_patterns_fail_when_llm_default_model_is_empty() -> None:
    empty_client = _EmptyDefaultModelClient()
    with pytest.raises(ValueError, match=r"default_model\(\) returned an empty model id"):
        PlannerExecutorPattern(
            llm_client=empty_client,
            tool_runtime=Toolbox(),
        ).run("Compute 1 + 1")

    with pytest.raises(ValueError, match=r"default_model\(\) returned an empty model id"):
        ReflexionPattern(
            llm_client=empty_client,
            tool_runtime=Toolbox(),
        ).run("Draft")

    with pytest.raises(ValueError, match=r"default_model\(\) returned an empty model id"):
        ConversationPattern(
            llm_client_a=empty_client,
            max_turns=1,
        ).run("Discuss this briefly.")
