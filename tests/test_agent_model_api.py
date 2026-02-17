from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest

from design_research_agents.agent import (
    AgentRuntime,
    MultiStepCodeToolCallingAgent,
    MultiStepDirectLLMAgent,
    MultiStepJsonToolCallingAgent,
    MultiStepToolRouterAgent,
    SingleStepCodeToolCallingAgent,
    SingleStepDirectLLMAgent,
    SingleStepJsonToolCallingAgent,
    SingleStepRouterAgent,
    SingleStepToolRouterAgent,
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
        SingleStepToolRouterAgent,
        SingleStepRouterAgent,
        SingleStepJsonToolCallingAgent,
        SingleStepCodeToolCallingAgent,
        MultiStepDirectLLMAgent,
        MultiStepToolRouterAgent,
        MultiStepJsonToolCallingAgent,
        MultiStepCodeToolCallingAgent,
        AgentRuntime,
    )
    for cls in classes:
        assert "model" not in inspect.signature(cls.__init__).parameters


def test_agent_constructor_signatures_expose_new_prompt_kwargs() -> None:
    direct_params = inspect.signature(SingleStepDirectLLMAgent.__init__).parameters
    assert "system_prompt" in direct_params
    assert "default_system_prompt" not in direct_params

    router_params = inspect.signature(SingleStepToolRouterAgent.__init__).parameters
    assert "user_prompt_template" in router_params
    assert "allowed_routes" in router_params

    json_params = inspect.signature(SingleStepJsonToolCallingAgent.__init__).parameters
    assert "allowed_tools" in json_params

    code_params = inspect.signature(SingleStepCodeToolCallingAgent.__init__).parameters
    assert "alternatives_prompt_target" in code_params

    multi_json_params = inspect.signature(MultiStepJsonToolCallingAgent.__init__).parameters
    assert "continuation_user_prompt_template" in multi_json_params
    assert "step_memory_tail_items" in multi_json_params

    multi_direct_params = inspect.signature(MultiStepDirectLLMAgent.__init__).parameters
    assert "controller_user_prompt_template" in multi_direct_params
    assert "step_memory_tail_items" in multi_direct_params

    multi_router_params = inspect.signature(MultiStepToolRouterAgent.__init__).parameters
    assert "user_prompt_template" in multi_router_params
    assert "stop_on_step_failure" in multi_router_params

    multi_code_params = inspect.signature(MultiStepCodeToolCallingAgent.__init__).parameters
    assert "continuation_system_prompt" in multi_code_params
    assert "continuation_memory_tail_items" in multi_code_params

    runtime_params = inspect.signature(AgentRuntime.__init__).parameters
    assert "mode" in runtime_params
    assert "controls" in runtime_params
    assert "tracer" in runtime_params
    assert "plan_execute_planner_system_prompt" not in runtime_params
    assert "propose_critic_critic_user_prompt_template" not in runtime_params
    assert "agent_routing_router_user_prompt_template" not in runtime_params


def test_direct_llm_agent_fails_when_llm_default_model_is_empty() -> None:
    with pytest.raises(ValueError, match=r"default_model\(\) returned an empty model id"):
        SingleStepDirectLLMAgent(llm_client=_EmptyDefaultModelClient()).run("Hello")


def test_agent_runtime_fails_when_llm_default_model_is_empty() -> None:
    with pytest.raises(ValueError, match=r"default_model\(\) returned an empty model id"):
        AgentRuntime(
            llm_client=_EmptyDefaultModelClient(),
            tool_runtime=Toolbox(),
            mode="react",
        ).run("Compute 1 + 1")


def test_agent_runtime_rejects_non_react_mode() -> None:
    with pytest.raises(ValueError, match="mode='react' only"):
        AgentRuntime(
            llm_client=_EmptyDefaultModelClient(),
            tool_runtime=Toolbox(),
            mode="plan_execute",
        )
