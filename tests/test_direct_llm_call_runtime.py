from __future__ import annotations

from collections.abc import Sequence

from design_research_agents.agent import DirectLLMCall
from design_research_agents.contracts.llm import LLMMessage, LLMRequest, LLMResponse
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
