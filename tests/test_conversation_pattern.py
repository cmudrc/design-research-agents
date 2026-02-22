from __future__ import annotations

from collections.abc import Sequence

import pytest

from design_research_agents._contracts._llm import LLMChatParams, LLMMessage, LLMResponse
from design_research_agents.patterns import ConversationPattern


class _CaptureLLMClient:
    def __init__(self, *, response_texts: Sequence[str], model: str = "capture-model") -> None:
        # Queue deterministic responses to validate transcript assembly exactly.
        self._responses = list(response_texts)
        self._model = model
        self.calls: list[dict[str, object]] = []

    def default_model(self) -> str:
        return self._model

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        # Capture full call payload so assertions can inspect prompt/template rendering.
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "params": params,
            }
        )
        if not self._responses:
            raise AssertionError("No stubbed responses remaining.")
        return LLMResponse(model=model, text=self._responses.pop(0), provider="capture")


def test_conversation_pattern_supports_distinct_clients_and_prompt_templates() -> None:
    speaker_a_client = _CaptureLLMClient(response_texts=["a1", "a2"], model="model-a")
    speaker_b_client = _CaptureLLMClient(response_texts=["b1", "b2"], model="model-b")
    pattern = ConversationPattern(
        llm_client_a=speaker_a_client,
        llm_client_b=speaker_b_client,
        max_turns=2,
        speaker_a_name="Researcher",
        speaker_b_name="Reviewer",
        speaker_a_system_prompt="You are Researcher A.",
        speaker_a_user_prompt_template="\n".join(
            [
                "A-turn=$turn",
                "Speaker=$speaker_name Partner=$partner_name",
                "Latest from $partner_name: $partner_message",
                "History: $conversation_transcript_json",
                "Task: $task_prompt",
            ]
        ),
        speaker_b_system_prompt="You are Reviewer B.",
        speaker_b_user_prompt_template="\n".join(
            [
                "B-turn=$turn",
                "Speaker=$speaker_name Partner=$partner_name",
                "Latest from $partner_name: $partner_message",
                "History: $conversation_transcript_json",
                "Task: $task_prompt",
            ]
        ),
        default_request_id_prefix="conversation-test",
    )

    result = pattern.run("Evaluate two migration options.")

    assert result.success
    assert pattern.workflow is not None
    assert result.output["participants"]["speaker_a"] == "Researcher"
    assert result.output["participants"]["speaker_b"] == "Reviewer"
    assert result.output["turns_executed"] == 2
    assert result.output["terminated_reason"] == "completed"
    transcript = result.output["transcript"]
    assert isinstance(transcript, list)
    assert len(transcript) == 4
    assert transcript[0]["speaker"] == "Researcher"
    assert transcript[1]["speaker"] == "Reviewer"
    assert transcript[-1]["message"] == "b2"
    assert result.output["final_output"]["speaker"] == "Reviewer"
    assert result.output["final_output"]["message"] == "b2"
    assert str(result.metadata["request_id"]).startswith("conversation-test:")

    assert len(speaker_a_client.calls) == 2
    assert len(speaker_b_client.calls) == 2

    # Validate that template variables are populated from partner message state.
    speaker_a_first_messages = speaker_a_client.calls[0]["messages"]
    assert isinstance(speaker_a_first_messages, list)
    assert speaker_a_first_messages[0].content == "You are Researcher A."
    assert "Latest from Reviewer: (none)" in speaker_a_first_messages[1].content

    speaker_b_first_messages = speaker_b_client.calls[0]["messages"]
    assert isinstance(speaker_b_first_messages, list)
    assert speaker_b_first_messages[0].content == "You are Reviewer B."
    assert "Latest from Researcher: a1" in speaker_b_first_messages[1].content


def test_conversation_pattern_defaults_speaker_b_client_to_speaker_a_client() -> None:
    shared_client = _CaptureLLMClient(response_texts=["a1", "b1", "a2", "b2"])
    pattern = ConversationPattern(
        llm_client_a=shared_client,
        max_turns=2,
    )

    result = pattern.run("Build a concise rollout recommendation.")

    assert result.success
    assert len(shared_client.calls) == 4
    assert len(result.output["transcript"]) == 4
    assert result.output["final_output"]["message"] == "b2"


def test_conversation_pattern_validates_constructor_and_template_overrides() -> None:
    # Constructor guardrails reject invalid turn counts and blank participant names.
    with pytest.raises(ValueError, match="max_turns"):
        ConversationPattern(llm_client_a=_CaptureLLMClient(response_texts=["x"]), max_turns=0)

    with pytest.raises(ValueError, match="speaker_a_name"):
        ConversationPattern(
            llm_client_a=_CaptureLLMClient(response_texts=["x"]),
            speaker_a_name="   ",
        )

    pattern = ConversationPattern(
        llm_client_a=_CaptureLLMClient(response_texts=["a1", "b1"]),
        max_turns=1,
        speaker_a_user_prompt_template="$missing_variable",
    )
    result = pattern.run("Trigger missing template variable.")
    assert not result.success
    assert result.output["terminated_reason"] == "speaker_a_failed"
    assert result.output["error"] == "Speaker A failed during conversation turn."
