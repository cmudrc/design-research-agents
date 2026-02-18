from __future__ import annotations

import json
from collections.abc import Sequence

from design_research_agents.contracts.llm import LLMChatParams, LLMMessage, LLMResponse
from design_research_agents.contracts.termination import (
    SOURCE_GUARDRAIL,
    SOURCE_INVALID_PAYLOAD,
    SOURCE_MODEL,
)
from design_research_agents.implementations.shared.agent_internal.multi_step_continuation import (
    llm_should_continue,
)
from design_research_agents.implementations.shared.agent_internal.response_schemas import (
    build_continuation_response_schema,
)


class _SequenceLLMClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.chat_calls = 0

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


def _continuation_call(
    llm_client: _SequenceLLMClient,
    *,
    memory: list[dict[str, object]],
    step_index: int,
) -> tuple[bool, str, str, LLMResponse | None]:
    return llm_should_continue(
        llm_client=llm_client,
        prompt="task prompt",
        memory=memory,
        step_index=step_index,
        max_steps=4,
        model="test-model",
        alternatives_prompt_target="user",
        alternatives_text="tool_a\ntool_b",
        retrieved_context="",
        continuation_system_prompt="Decide whether to continue.",
        continuation_user_prompt_template=(
            "Step {step_number}\n"
            "Task: {task_prompt}\n"
            "Memory: {memory_tail}\n"
            "Context: {retrieved_context}"
        ),
        continuation_response_schema=build_continuation_response_schema(),
        continuation_memory_tail_items=4,
        alternatives_section_label="Alternatives",
        agent_name="MultiStepJsonToolCallingAgent",
    )


def test_llm_should_continue_accepts_strict_payload() -> None:
    llm_client = _SequenceLLMClient([json.dumps({"continue": True, "thought": "continue"})])

    should_continue, reason, source, response = _continuation_call(
        llm_client,
        memory=[{"kind": "task", "prompt": "task prompt"}],
        step_index=0,
    )

    assert should_continue is True
    assert reason == "continue"
    assert source == SOURCE_MODEL
    assert response is not None
    assert llm_client.chat_calls == 1


def test_llm_should_continue_invalid_payload_is_terminal_false() -> None:
    llm_client = _SequenceLLMClient([json.dumps({"decision": "CONTINUE"})])

    should_continue, reason, source, _ = _continuation_call(
        llm_client,
        memory=[{"kind": "task", "prompt": "task prompt"}],
        step_index=1,
    )

    assert should_continue is False
    assert reason == "invalid continuation payload"
    assert source == SOURCE_INVALID_PAYLOAD


def test_llm_should_continue_first_step_guardrail_overrides_false_without_observation() -> None:
    llm_client = _SequenceLLMClient([json.dumps({"continue": False, "thought": "stop"})])

    should_continue, reason, source, _ = _continuation_call(
        llm_client,
        memory=[{"kind": "task", "prompt": "task prompt"}],
        step_index=0,
    )

    assert should_continue is True
    assert reason == "first-step guardrail"
    assert source == SOURCE_GUARDRAIL


def test_llm_should_continue_no_guardrail_after_observation() -> None:
    llm_client = _SequenceLLMClient([json.dumps({"continue": False, "thought": "stop"})])

    should_continue, reason, source, _ = _continuation_call(
        llm_client,
        memory=[
            {"kind": "task", "prompt": "task prompt"},
            {"kind": "observation", "step": 1, "success": True},
        ],
        step_index=0,
    )

    assert should_continue is False
    assert reason == "stop"
    assert source == SOURCE_MODEL
