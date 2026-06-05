from __future__ import annotations

import pytest

from design_research_agents._contracts._llm import LLMChatParams, LLMMessage, LLMRequest, LLMResponse
from design_research_agents._contracts._memory import MemoryWriteRecord
from design_research_agents._contracts._workflow import DelegateBatchCall, ModelStep
from design_research_agents._runtime._workflow._executors import _common as executor_common


class _GenerateClient:
    def __init__(self, response: object) -> None:
        self.response = response

    def generate(self, request: LLMRequest) -> object:
        del request
        return self.response


class _ChatOnlyClient:
    def __init__(self, response: object = LLMResponse(text="chat")) -> None:
        self.response = response
        self.calls: list[tuple[tuple[LLMMessage, ...], str, LLMChatParams]] = []

    def default_model(self) -> str:
        return "default-chat-model"

    def chat(
        self,
        messages: tuple[LLMMessage, ...],
        *,
        model: str,
        params: LLMChatParams,
    ) -> object:
        self.calls.append((messages, model, params))
        return self.response


class _NoModelClient:
    pass


def _request(**overrides: object) -> LLMRequest:
    values = {"messages": (LLMMessage(role="user", content="hello"),), **overrides}
    return LLMRequest(**values)


def test_model_step_private_helpers_cover_schema_and_validation_branches() -> None:
    request = _request()
    terminal_context = {
        "_workflow": {
            "is_terminal_step": True,
            "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        }
    }

    assert executor_common._apply_terminal_output_schema(
        llm_request=request,
        step_context={},
    ) is request
    assert executor_common._apply_terminal_output_schema(
        llm_request=request,
        step_context={"_workflow": {"is_terminal_step": False}},
    ) is request
    with_explicit_schema = _request(response_schema={"type": "object"})
    assert (
        executor_common._apply_terminal_output_schema(
            llm_request=with_explicit_schema,
            step_context=terminal_context,
        )
        is with_explicit_schema
    )
    injected = executor_common._apply_terminal_output_schema(
        llm_request=request,
        step_context=terminal_context,
    )
    assert injected.response_schema == terminal_context["_workflow"]["output_schema"]

    with pytest.raises(TypeError, match="request_builder"):
        executor_common._build_model_request(
            step=ModelStep(
                step_id="bad_request",
                llm_client=_GenerateClient(LLMResponse(text="unused")),
                request_builder=lambda context: {"not": "a request"},
            ),
            step_context={},
        )

    with pytest.raises(TypeError, match=r"request\.model"):
        executor_common._resolve_model_request_model_id(llm_client=_NoModelClient(), llm_request=request)

    with pytest.raises(TypeError, match="generate"):
        executor_common._execute_model_request(
            step=ModelStep(
                step_id="bad_generate",
                llm_client=_GenerateClient({"bad": True}),
                request_builder=lambda context: request,
            ),
            llm_request=request,
        )

    with pytest.raises(TypeError, match="generate"):
        executor_common._execute_model_request(
            step=ModelStep(
                step_id="missing_client",
                llm_client=object(),
                request_builder=lambda context: request,
            ),
            llm_request=request,
        )

    with pytest.raises(TypeError, match="chat"):
        executor_common._execute_model_request(
            step=ModelStep(
                step_id="bad_chat",
                llm_client=_ChatOnlyClient(response={"bad": True}),
                request_builder=lambda context: request,
            ),
            llm_request=request,
        )

    chat_client = _ChatOnlyClient()
    chat_response = executor_common._execute_model_request(
        step=ModelStep(
            step_id="chat",
            llm_client=chat_client,
            request_builder=lambda context: request,
        ),
        llm_request=_request(model=None, temperature=0.2, max_tokens=12, provider_options={"x": 1}),
    )
    assert chat_response == LLMResponse(text="chat")
    assert chat_client.calls[0][1] == "default-chat-model"
    assert chat_client.calls[0][2].provider_options == {"x": 1}


def test_model_response_private_helpers_cover_parser_and_final_output_fallbacks() -> None:
    request = _request()
    parser_step = ModelStep(
        step_id="parse",
        llm_client=_GenerateClient(LLMResponse(text="unused")),
        request_builder=lambda context: request,
        response_parser=lambda response, context: ["not", "a", "mapping"],
    )

    with pytest.raises(TypeError, match="response_parser"):
        executor_common._parse_model_response_payload(
            step=parser_step,
            model_response=LLMResponse(text="raw"),
            step_context={},
        )

    default_step = ModelStep(
        step_id="default_parse",
        llm_client=_GenerateClient(LLMResponse(text="unused")),
        request_builder=lambda context: request,
    )
    assert executor_common._parse_model_response_payload(
        step=default_step,
        model_response=LLMResponse(text="raw text"),
        step_context={},
    ) == {"model_text": "raw text"}
    assert executor_common._resolve_model_final_output(
        parsed_payload={"final_output": {"answer": 1}},
        model_response=LLMResponse(text="ignored"),
    ) == {"answer": 1}
    assert executor_common._resolve_model_final_output(
        parsed_payload={"answer": 2},
        model_response=LLMResponse(text="ignored"),
    ) == {"answer": 2}
    assert executor_common._resolve_model_final_output(
        parsed_payload={},
        model_response=LLMResponse(text="fallback"),
    ) == {"model_text": "fallback"}


def test_delegate_batch_private_helpers_cover_validation_edges() -> None:
    call = DelegateBatchCall(call_id="direct", delegate=object(), prompt="do it")
    assert executor_common._normalize_delegate_batch_call(call, index=1) is call

    with pytest.raises(TypeError, match="sequence"):
        executor_common._normalize_delegate_batch_calls(raw_calls="bad")
    with pytest.raises(ValueError, match="Duplicate"):
        executor_common._normalize_delegate_batch_calls(
            raw_calls=[
                {"call_id": "dup", "delegate": object(), "prompt": "one"},
                {"call_id": "dup", "delegate": object(), "prompt": "two"},
            ]
        )
    for raw_call, expected in (
        (object(), "DelegateBatchStep calls"),
        ({"call_id": " ", "delegate": object(), "prompt": "x"}, "call_id"),
        ({"call_id": "missing_delegate", "prompt": "x"}, "missing delegate"),
        ({"call_id": "missing_prompt", "delegate": object()}, "missing prompt"),
        ({"call_id": "empty_prompt", "delegate": object(), "prompt": " "}, "prompt"),
    ):
        with pytest.raises((TypeError, ValueError), match=expected):
            executor_common._normalize_delegate_batch_call(raw_call, index=1)

    normalized = executor_common._normalize_delegate_batch_call(
        {
            "delegate": object(),
            "prompt": "  hello  ",
            "execution_mode": "unknown",
            "failure_policy": "unknown",
        },
        index=3,
    )
    assert normalized.call_id == "call_3"
    assert normalized.prompt == "hello"
    assert normalized.execution_mode == "sequential"
    assert normalized.failure_policy == "skip_dependents"

    assert executor_common._resolve_delegate_batch_final_output([]) == {}
    assert executor_common._resolve_delegate_batch_final_output([{"output": "bad"}]) == {}
    assert executor_common._resolve_delegate_batch_final_output([{"output": {"final_output": {"answer": 1}}}]) == {
        "answer": 1
    }
    assert executor_common._resolve_delegate_batch_final_output([{"output": {"raw": True}}]) == {"raw": True}


def test_memory_write_record_private_helper_covers_payload_edges() -> None:
    direct = MemoryWriteRecord(content="direct")
    normalized = executor_common._normalize_memory_write_records(
        [
            direct,
            "plain",
            {"content": 42, "metadata": {"kind": "number"}, "item_id": " item-1 "},
        ]
    )

    assert normalized[0] is direct
    assert normalized[1].content == "plain"
    assert normalized[2].content == "42"
    assert normalized[2].metadata == {"kind": "number"}
    assert normalized[2].item_id == "item-1"
    with pytest.raises(TypeError, match="sequence"):
        executor_common._normalize_memory_write_records("bad")
    with pytest.raises(ValueError, match="content"):
        executor_common._normalize_memory_write_records([{"metadata": {}}])
    with pytest.raises(TypeError, match="Unsupported"):
        executor_common._normalize_memory_write_records([object()])
