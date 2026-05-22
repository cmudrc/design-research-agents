"""Tests for reusable JSON prompt-workflow construction."""

from __future__ import annotations

from collections.abc import Iterator

from design_research_agents import LLMMessage, LLMRequest, LLMResponse
from design_research_agents._contracts._llm import LLMDelta
from design_research_agents.workflow import build_json_prompt_workflow


class _FakeLLMClient:
    """Small request-capturing LLM client for workflow tests."""

    def __init__(self, response_text: str) -> None:
        """Store one response text and initialize captured requests."""
        self.response_text = response_text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Capture the request and return the configured response."""
        self.requests.append(request)
        return LLMResponse(
            text=self.response_text,
            model="fake-model",
            provider="fake-provider",
            usage={"prompt_tokens": 11, "completion_tokens": 7},
        )

    def stream(self, _request: LLMRequest) -> Iterator[LLMDelta]:
        """Return an empty stream for protocol completeness."""
        return iter(())


def test_build_json_prompt_workflow_parses_fenced_json_and_usage() -> None:
    """The helper should emit parsed final output, events, and usage metrics."""
    client = _FakeLLMClient('```json\n{"choice": 3, "label": "compact"}\n```')
    workflow = build_json_prompt_workflow(
        llm_client=client,
        response_schema={
            "type": "object",
            "properties": {"choice": {"type": "number"}, "label": {"type": "string"}},
            "required": ["choice", "label"],
        },
        request_metadata={"study_id": "study-1"},
        default_request_id_prefix="test-json",
    )

    result = workflow.run("Return one compact JSON candidate.", request_id="run-1")
    parsed = result.step_results["json_response"].output["parsed"]

    assert result.success is True
    assert result.output["final_output"] == {"choice": 3, "label": "compact"}
    assert parsed["metrics"] == {
        "cost_usd": 0.0,
        "input_tokens": 11,
        "output_tokens": 7,
    }
    assert parsed["events"][0]["event_type"] == "model_response"
    assert client.requests[0].messages == (
        LLMMessage(
            role="system",
            content=(
                "You are a careful study participant. Return valid JSON only and match the requested schema exactly."
            ),
        ),
        LLMMessage(role="user", content="Return one compact JSON candidate."),
    )
    assert client.requests[0].metadata == {"study_id": "study-1"}


def test_build_json_prompt_workflow_rejects_non_json_response() -> None:
    """Invalid JSON should fail the workflow step loudly."""
    workflow = build_json_prompt_workflow(
        llm_client=_FakeLLMClient("not json"),
        response_schema={"type": "object"},
    )

    result = workflow.run("Return JSON.", request_id="run-2")

    assert result.success is False
    assert "valid JSON" in str(result.step_results["json_response"].error)
