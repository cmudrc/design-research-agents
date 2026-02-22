from __future__ import annotations

from collections.abc import Sequence

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMMessage, LLMRequest, LLMResponse
from design_research_agents._contracts._workflow import WorkflowStepResult
from design_research_agents._implementations._agents import _direct_llm_call as direct_llm_impl
from design_research_agents.agent import DirectLLMCall
from tests.helpers.workflow_stubs import SequenceLLMClient


class _CaptureGenerateClient:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def default_model(self) -> str:
        return "capture-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            model=request.model or self.default_model(),
            text="captured",
            provider="capture",
        )


class _RaisingGenerateClient:
    def default_model(self) -> str:
        return "raising-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        del request
        raise RuntimeError("boom")


def test_direct_llm_call_returns_workflow_first_result() -> None:
    agent = DirectLLMCall(llm_client=SequenceLLMClient(response_texts=["hello"]))

    result = agent.run(
        "Say hello.",
        request_id="req-direct-001",
        dependencies={"beta": 2, "alpha": 1},
    )

    assert result.success is True
    assert result.output["model_text"] == "hello"
    assert result.output["final_output"] == "hello"
    assert isinstance(result.output["workflow"], dict)
    assert isinstance(result.output["artifacts"], list)
    assert result.metadata["request_id"] == "req-direct-001"
    assert result.metadata["dependency_keys"] == ["alpha", "beta"]
    assert result.metadata["llm_call"]["source"] == "direct"
    assert result.model_response is not None
    assert result.model_response.text == "hello"
    assert result.tool_results == []


def test_direct_llm_call_builds_expected_llm_request_metadata() -> None:
    llm_client = _CaptureGenerateClient()
    agent = DirectLLMCall(
        llm_client=llm_client,
        system_prompt="You are concise.",
        temperature=0.2,
        max_tokens=64,
        provider_options={"seed": 7},
    )

    result = agent.run("Summarize this.")

    assert result.success is True
    assert len(llm_client.requests) == 1
    request = llm_client.requests[0]
    assert request.model == "capture-model"
    assert request.temperature == 0.2
    assert request.max_tokens == 64
    assert request.provider_options == {"seed": 7}
    assert request.metadata["agent"] == "DirectLLMCall"
    assert request.metadata["message_source"] == "prompt"
    assert request.metadata["request_id"]
    assert isinstance(request.messages, Sequence)
    assert list(request.messages) == [
        LLMMessage(role="system", content="You are concise."),
        LLMMessage(role="user", content="Summarize this."),
    ]


def test_direct_llm_call_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        DirectLLMCall(llm_client=_CaptureGenerateClient(), max_tokens=0)


def test_direct_llm_call_prepare_request_step_validation_errors() -> None:
    agent = DirectLLMCall(llm_client=_CaptureGenerateClient())

    with pytest.raises(TypeError, match="schema input mapping"):
        agent._prepare_request_step({})

    with pytest.raises(TypeError, match="normalized_input must be a mapping"):
        agent._prepare_request_step({"inputs": {"normalized_input": "not-a-mapping"}})


def test_direct_llm_call_dependency_helpers_cover_fallbacks() -> None:
    assert direct_llm_impl._dependency_output(context={}, step_id="prepare_request") == {}
    assert direct_llm_impl._dependency_output(
        context={"dependency_results": {"prepare_request": {"output": {"k": 1}}}},
        step_id="prepare_request",
    ) == {"k": 1}
    assert (
        direct_llm_impl._dependency_output(
            context={"dependency_results": {"prepare_request": {"output": "not-mapping"}}},
            step_id="prepare_request",
        )
        == {}
    )

    assert direct_llm_impl._int_or_default(True, default=9) == 1
    assert direct_llm_impl._int_or_default(7, default=9) == 7
    assert direct_llm_impl._int_or_default("8", default=9) == 8
    assert direct_llm_impl._int_or_default("bad", default=9) == 9
    assert direct_llm_impl._int_or_default(object(), default=9) == 9


def test_direct_llm_call_workflow_failure_raiser_paths() -> None:
    with pytest.raises(ValueError, match="bad-step"):
        direct_llm_impl._raise_workflow_failure(
            ExecutionResult(
                success=False,
                output={},
                execution_order=["failed_step"],
                step_results={
                    "failed_step": WorkflowStepResult(
                        step_id="failed_step",
                        status="failed",
                        success=False,
                        error="bad-step",
                    )
                },
            )
        )

    with pytest.raises(RuntimeError, match="failed_step"):
        direct_llm_impl._raise_workflow_failure(
            ExecutionResult(
                success=False,
                output={},
                execution_order=["failed_step"],
                step_results={
                    "failed_step": WorkflowStepResult(
                        step_id="failed_step",
                        status="failed",
                        success=False,
                        error=None,
                    )
                },
            )
        )

    with pytest.raises(RuntimeError, match="execution failed"):
        direct_llm_impl._raise_workflow_failure(
            ExecutionResult(
                success=False,
                output={},
                execution_order=[],
                step_results={},
            )
        )


def test_direct_llm_call_call_and_finalize_step_validation_paths() -> None:
    agent = DirectLLMCall(llm_client=_RaisingGenerateClient())
    with pytest.raises(TypeError, match="Prepared request payload is invalid"):
        agent._call_model_step({"dependency_results": {"prepare_request": {"output": {"resolved_model": 7}}}})

    with pytest.raises(RuntimeError, match="boom"):
        agent._call_model_step(
            {
                "dependency_results": {
                    "prepare_request": {
                        "output": {
                            "resolved_model": "raising-model",
                            "llm_request": LLMRequest(
                                messages=[LLMMessage(role="user", content="x")],
                                model="raising-model",
                            ),
                            "messages": [],
                            "message_source": "prompt",
                        }
                    }
                }
            }
        )

    with pytest.raises(TypeError, match="missing LLM request/response payload"):
        agent._finalize_step({"dependency_results": {}})
