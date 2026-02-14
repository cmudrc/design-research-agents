"""Contract and agent behavior tests for protocol compliance and integration flow."""

import json
from collections.abc import Iterator

import pytest

from design_research_agents.agent import (
    MultiStepAgent,
    RouterAgent,
    SingleStepCodeAgent,
    ToolCallingAgent,
)
from design_research_agents.contracts import (
    Agent,
    LLMAuthError,
    LLMChatParams,
    LLMClient,
    LLMInvalidRequestError,
    LLMMessage,
    LLMResponse,
    LLMStreamEvent,
    ToolRuntime,
)
from design_research_agents.llm import (
    BaseLLMClient,
    configure_llama_cpp_server,
    configure_openai,
    shutdown_llama_cpp_server,
)
from design_research_agents.tools import BaseToolRuntime


def _requires_llm_client(client: LLMClient) -> None:
    """Mypy helper ensuring protocol compatibility."""


def _requires_tool_runtime(runtime: ToolRuntime) -> None:
    """Mypy helper ensuring protocol compatibility."""


def _requires_agent(agent: Agent) -> None:
    """Mypy helper ensuring protocol compatibility."""


def test_llm_chat_params_support_schema_and_provider_options() -> None:
    params = LLMChatParams(
        temperature=0.1,
        max_tokens=256,
        response_schema={"$id": "urn:test:base"},
        provider_options={"reasoning_effort": "low"},
    )
    assert params.temperature == 0.1
    assert params.max_tokens == 256
    assert params.response_schema == {"$id": "urn:test:base"}
    assert params.provider_options["reasoning_effort"] == "low"


def test_base_llm_client_uses_existing_backend_path() -> None:
    llm_client = BaseLLMClient(backend="echo-test")
    response = llm_client.chat(
        messages=[
            LLMMessage(role="system", content="sys"),
            LLMMessage(role="user", content="hello world"),
        ],
        model="base-model",
        params=LLMChatParams(
            response_schema={"$id": "urn:test:schema"},
            provider_options={"reasoning_effort": "low"},
        ),
    )
    assert response.model == "base-model"
    assert response.text.startswith("[echo-test]")
    assert response.provider == "echo-test"
    assert response.raw_output is not None
    assert response.raw_output["backend"] == "echo-test"
    assert response.raw_output["requested_model"] == "base-model"
    assert response.raw_output["response_schema"] == {"$id": "urn:test:schema"}


def test_base_llm_client_schema_hint_uses_compact_instruction() -> None:
    llm_client = BaseLLMClient(backend="echo-test")
    response = llm_client.chat(
        messages=[LLMMessage(role="user", content="Give me a quick status update.")],
        model="base-model",
        params=LLMChatParams(
            response_schema={
                "type": "object",
                "required": ["summary", "risks"],
                "properties": {
                    "summary": {"type": "string"},
                    "risks": {"type": "array"},
                },
            },
        ),
    )

    assert "Return JSON matching this schema" not in response.text
    assert '{"properties"' not in response.text
    assert "Required top-level keys: summary, risks." in response.text
    assert "Field expectations: summary (string); risks (array)." in response.text


def test_base_llm_stream_emits_delta_then_completed() -> None:
    llm_client = BaseLLMClient(backend="echo-test")
    stream_events = list(
        llm_client.stream_chat(
            messages=[LLMMessage(role="user", content="stream me")],
            model="stream-model",
            params=LLMChatParams(),
        )
    )
    assert [event.kind for event in stream_events] == ["delta", "completed"]
    assert stream_events[0].delta_text is not None
    assert stream_events[1].response is not None


def test_base_runtime_and_agent_satisfy_protocols() -> None:
    llm_client = BaseLLMClient(backend="echo-test")
    tool_runtime = BaseToolRuntime()
    agent = RouterAgent(llm_client=llm_client, tool_runtime=tool_runtime)
    _requires_llm_client(llm_client)
    _requires_tool_runtime(tool_runtime)
    _requires_agent(agent)

    result = agent.run(
        {
            "prompt": "Calculate 6 * 7",
            "alternatives": [
                {
                    "tool_name": "text_stats_tool",
                    "description": "Analyze and summarize text content.",
                },
                {
                    "tool_name": "calculator_tool",
                    "description": "Evaluate arithmetic expressions.",
                    "tool_input": {"expression": "6 * 7"},
                },
            ],
        },
        {},
    )
    assert result.success
    assert result.tool_results[0].tool_name == "calculator_tool"
    assert result.tool_results[0].output["result"] == 42.0


class _StaticResponseLLMClient:
    """Deterministic LLM client stub used by tool-calling agent tests."""

    def __init__(self, *, response_text: str) -> None:
        self._response_text = response_text

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        return LLMResponse(
            model=model,
            text=self._response_text,
            provider="test-stub",
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        del messages, params
        response = LLMResponse(
            model=model,
            text=self._response_text,
            provider="test-stub",
        )
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)


class _SequenceResponseLLMClient:
    """Deterministic LLM client stub that returns a configured response sequence."""

    def __init__(self, *, response_texts: list[str]) -> None:
        self._responses = list(response_texts)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        if not self._responses:
            raise AssertionError("No more stubbed LLM responses available.")
        return LLMResponse(
            model=model,
            text=self._responses.pop(0),
            provider="test-sequence-stub",
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)


def test_tool_calling_agent_executes_model_selected_tool_with_args() -> None:
    llm_client = _StaticResponseLLMClient(
        response_text=json.dumps(
            {
                "tool_name": "calculator_tool",
                "tool_input": {"expression": "6 * 7"},
                "reason": "This is an arithmetic request.",
            }
        )
    )
    tool_runtime = BaseToolRuntime()
    agent = ToolCallingAgent(llm_client=llm_client, tool_runtime=tool_runtime)
    _requires_agent(agent)

    result = agent.run(
        input={
            "prompt": "Calculate 6 * 7",
            "tools": [
                {
                    "tool_name": "text_stats_tool",
                    "description": "Analyze and summarize text content.",
                },
                {
                    "tool_name": "calculator_tool",
                    "description": "Evaluate arithmetic expressions.",
                },
            ],
        },
        context={},
    )
    assert result.success
    assert result.tool_results[0].tool_name == "calculator_tool"
    assert result.tool_results[0].output["result"] == 42.0
    assert result.metadata["tool_call"]["source"] == "model"


def test_tool_calling_agent_falls_back_when_model_output_is_not_json() -> None:
    llm_client = _StaticResponseLLMClient(response_text="not-json")
    tool_runtime = BaseToolRuntime()
    agent = ToolCallingAgent(llm_client=llm_client, tool_runtime=tool_runtime)

    result = agent.run(
        input={
            "prompt": "Count words in this exact sentence.",
            "analysis_text": "one two three four",
            "tools": [
                {
                    "tool_name": "calculator_tool",
                    "description": "Evaluate arithmetic expressions.",
                },
                {
                    "tool_name": "text_stats_tool",
                    "description": "Compute text statistics and word counts.",
                },
            ],
        },
        context={},
    )
    assert result.success
    assert result.tool_results[0].tool_name == "text_stats_tool"
    assert result.tool_results[0].output["word_count"] == 4
    assert result.metadata["tool_call"]["source"] == "fallback"


def test_single_step_code_agent_executes_multiple_tool_calls() -> None:
    llm_client = _StaticResponseLLMClient(
        response_text="\n".join(
            [
                'calc = call_tool("calculator_tool", {"expression": "6 * 7"})',
                'stats = call_tool("text_stats_tool", {"text": f"Result is {calc[\'result\']}."})',
                'final_output = {"result": calc["result"], "word_count": stats["word_count"]}',
            ]
        )
    )
    tool_runtime = BaseToolRuntime()
    agent = SingleStepCodeAgent(llm_client=llm_client, tool_runtime=tool_runtime)
    _requires_agent(agent)

    result = agent.run(
        input={
            "prompt": "Calculate 6 * 7 and summarize the text.",
            "tools": [
                {"tool_name": "calculator_tool"},
                {"tool_name": "text_stats_tool"},
            ],
        },
        context={},
    )
    assert result.success
    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "calculator_tool",
        "text_stats_tool",
    ]
    assert result.output["final_output"]["result"] == 42.0
    assert result.output["final_output"]["word_count"] == 3


def test_single_step_code_agent_rejects_disallowed_tool_call() -> None:
    llm_client = _StaticResponseLLMClient(
        response_text="\n".join(
            [
                'stats = call_tool("text_stats_tool", {"text": "blocked"})',
                'final_output = {"word_count": stats["word_count"]}',
            ]
        )
    )
    tool_runtime = BaseToolRuntime()
    agent = SingleStepCodeAgent(llm_client=llm_client, tool_runtime=tool_runtime)

    result = agent.run(
        input={
            "prompt": "Try to call disallowed tool.",
            "tools": [{"tool_name": "calculator_tool"}],
        },
        context={},
    )
    assert not result.success
    assert "allowed tool list" in result.output["error"]
    assert result.metadata["stage"] == "code_execution"


def test_single_step_code_agent_requires_final_output_dict() -> None:
    llm_client = _StaticResponseLLMClient(
        response_text="\n".join(
            [
                'call_tool("text_stats_tool", {"text": "hello world"})',
                'final_output = "not-a-dict"',
            ]
        )
    )
    tool_runtime = BaseToolRuntime()
    agent = SingleStepCodeAgent(llm_client=llm_client, tool_runtime=tool_runtime)

    result = agent.run(
        input={
            "prompt": "Return anything.",
            "tools": [{"tool_name": "text_stats_tool"}],
        },
        context={},
    )
    assert not result.success
    assert "final_output" in result.output["error"]


def test_single_step_code_agent_uses_last_tool_output_when_final_output_missing() -> None:
    llm_client = _StaticResponseLLMClient(
        response_text='call_tool("text_stats_tool", {"text": "hello world"})'
    )
    tool_runtime = BaseToolRuntime()
    agent = SingleStepCodeAgent(llm_client=llm_client, tool_runtime=tool_runtime)

    result = agent.run(
        input={
            "prompt": "Count words.",
            "tools": [{"tool_name": "text_stats_tool"}],
        },
        context={},
    )
    assert result.success
    assert result.output["final_output"]["word_count"] == 2


def test_single_step_code_agent_requires_at_least_one_tool_call() -> None:
    llm_client = _StaticResponseLLMClient(response_text='final_output = {"status": "done"}')
    tool_runtime = BaseToolRuntime()
    agent = SingleStepCodeAgent(llm_client=llm_client, tool_runtime=tool_runtime)

    result = agent.run(
        input={
            "prompt": "Return done.",
            "tools": [{"tool_name": "text_stats_tool"}],
        },
        context={},
    )
    assert not result.success
    assert "at least one tool" in result.output["error"]


def test_single_step_code_agent_schema_validation_is_optional() -> None:
    code = "\n".join(
        [
            'calc = call_tool("calculator_tool", {"expression": 123})',
            'final_output = {"result": calc["result"]}',
        ]
    )
    llm_client = _StaticResponseLLMClient(response_text=code)
    tool_runtime = BaseToolRuntime()
    agent = SingleStepCodeAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        validate_tool_input_schema=False,
    )

    non_validating_result = agent.run(
        input={
            "prompt": "Calculate 123.",
            "tools": [{"tool_name": "calculator_tool"}],
        },
        context={},
    )
    assert non_validating_result.success
    assert non_validating_result.output["final_output"]["result"] == 123.0

    validating_result = agent.run(
        input={
            "prompt": "Calculate 123.",
            "tools": [{"tool_name": "calculator_tool"}],
            "validate_tool_input_schema": True,
        },
        context={},
    )
    assert not validating_result.success
    assert "must be a string" in validating_result.output["error"]


def test_multi_step_agent_runs_two_action_observation_steps() -> None:
    llm_client = _SequenceResponseLLMClient(
        response_texts=[
            '{"continue": true, "reason": "Need arithmetic first."}',
            "\n".join(
                [
                    'calc = call_tool("calculator_tool", {"expression": "6 * 7"})',
                    'final_output = {"result": calc["result"]}',
                ]
            ),
            '{"continue": true, "reason": "Need summary stats next."}',
            "\n".join(
                [
                    'stats = call_tool("text_stats_tool", {"text": "Result is 42"})',
                    'final_output = {"word_count": stats["word_count"]}',
                ]
            ),
            '{"continue": false, "reason": "Done."}',
        ]
    )
    tool_runtime = BaseToolRuntime()
    agent = MultiStepAgent(llm_client=llm_client, tool_runtime=tool_runtime, max_steps=5)
    _requires_agent(agent)

    result = agent.run(
        input={
            "prompt": "Compute 6 * 7, then summarize the result text.",
            "tools": [
                {"tool_name": "calculator_tool"},
                {"tool_name": "text_stats_tool"},
            ],
        },
        context={},
    )
    assert result.success
    assert result.output["steps_executed"] == 2
    assert result.output["final_output"]["word_count"] == 3
    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "calculator_tool",
        "text_stats_tool",
    ]
    assert result.output["terminated_reason"] == "continuation_stopped:model"
    assert len(result.metadata["continuation"]) == 3


def test_multi_step_agent_uses_fallback_continuation_on_invalid_json() -> None:
    llm_client = _SequenceResponseLLMClient(
        response_texts=[
            "not-json",
            "\n".join(
                [
                    'stats = call_tool("text_stats_tool", {"text": "hello world"})',
                    'final_output = {"word_count": stats["word_count"]}',
                ]
            ),
            "still-not-json",
        ]
    )
    tool_runtime = BaseToolRuntime()
    agent = MultiStepAgent(llm_client=llm_client, tool_runtime=tool_runtime, max_steps=5)

    result = agent.run(
        input={
            "prompt": "Count words one time.",
            "tools": [{"tool_name": "text_stats_tool"}],
        },
        context={},
    )
    assert result.success
    assert result.output["steps_executed"] == 1
    assert result.output["terminated_reason"] == "continuation_stopped:fallback"
    continuation_trace = result.metadata["continuation"]
    assert continuation_trace[0]["source"] == "fallback"
    assert continuation_trace[1]["source"] == "fallback"


def test_multi_step_agent_runs_first_step_even_if_model_stops_immediately() -> None:
    llm_client = _SequenceResponseLLMClient(
        response_texts=[
            '{"continue": false, "reason": "Task completion"}',
            "\n".join(
                [
                    'calc = call_tool("calculator_tool", {"expression": "6 * 7"})',
                    'final_output = {"result": calc["result"]}',
                ]
            ),
            '{"continue": false, "reason": "Done."}',
        ]
    )
    tool_runtime = BaseToolRuntime()
    agent = MultiStepAgent(llm_client=llm_client, tool_runtime=tool_runtime, max_steps=3)

    result = agent.run(
        input={
            "prompt": "Compute 6 * 7.",
            "tools": [{"tool_name": "calculator_tool"}],
        },
        context={},
    )
    assert result.success
    assert result.output["steps_executed"] == 1
    assert result.output["final_output"]["result"] == 42.0
    continuation_trace = result.metadata["continuation"]
    assert continuation_trace[0]["source"] == "guardrail"
    assert continuation_trace[1]["source"] == "model"
    assert result.output["terminated_reason"] == "continuation_stopped:model"


def test_multi_step_agent_recovers_when_step_code_omits_final_output() -> None:
    llm_client = _SequenceResponseLLMClient(
        response_texts=[
            '{"continue": false, "reason": "Task completion"}',
            "\n".join(
                [
                    'call_tool("calculator_tool", {"expression": "6 * 7"})',
                    'call_tool("text_stats_tool", {"text": "Result is 42"})',
                ]
            ),
            '{"continue": false, "reason": "Done."}',
        ]
    )
    tool_runtime = BaseToolRuntime()
    agent = MultiStepAgent(llm_client=llm_client, tool_runtime=tool_runtime, max_steps=3)

    result = agent.run(
        input={
            "prompt": "Compute 6 * 7, then summarize text.",
            "tools": [
                {"tool_name": "calculator_tool"},
                {"tool_name": "text_stats_tool"},
            ],
        },
        context={},
    )
    assert result.success
    assert result.output["steps_executed"] == 1
    assert result.output["final_output"]["word_count"] == 3
    assert result.output["terminated_reason"] == "continuation_stopped:model"


def test_multi_step_agent_returns_structured_failure_when_step_fails() -> None:
    llm_client = _SequenceResponseLLMClient(
        response_texts=[
            '{"continue": true, "reason": "Do work."}',
            "\n".join(
                [
                    'bad = call_tool("text_stats_tool", {"text": "blocked"})',
                    'final_output = {"bad": bad}',
                ]
            ),
        ]
    )
    tool_runtime = BaseToolRuntime()
    agent = MultiStepAgent(llm_client=llm_client, tool_runtime=tool_runtime, max_steps=3)

    result = agent.run(
        input={
            "prompt": "Try forbidden tool.",
            "tools": [{"tool_name": "calculator_tool"}],
        },
        context={},
    )
    assert not result.success
    assert result.metadata["stage"] == "step_execution"
    assert result.output["terminated_reason"] == "step_failure"
    assert "allowed tool list" in result.output["error"]


def test_router_agent_selects_text_stats_from_explicit_alternatives() -> None:
    llm_client = BaseLLMClient(backend="echo-test")
    tool_runtime = BaseToolRuntime()
    agent = RouterAgent(llm_client=llm_client, tool_runtime=tool_runtime)

    result = agent.run(
        {
            "prompt": "Count words in this exact sentence.",
            "analysis_text": "one two three four",
            "alternatives": [
                {
                    "tool_name": "calculator_tool",
                    "description": "Evaluate arithmetic expressions.",
                },
                {
                    "tool_name": "text_stats_tool",
                    "description": "Compute text statistics and word counts.",
                },
            ],
        },
        {},
    )
    assert result.success
    assert result.tool_results[0].tool_name == "text_stats_tool"
    assert result.tool_results[0].output["word_count"] == 4
    routing = result.metadata["routing"]
    assert routing["selected_tool_name"] == "text_stats_tool"
    assert routing["alternatives"] == ["calculator_tool", "text_stats_tool"]


def test_base_llm_client_defaults_to_active_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Active backend set via configure_openai should be picked up by BaseLLMClient().
    def fake_openai_complete(prompt: str, **_: object) -> str:
        return f"openai:{prompt}"

    monkeypatch.setattr(
        "design_research_agents.llm.backends.adapters.openai_complete",
        fake_openai_complete,
    )

    configure_openai(
        model="gpt-default-route",
        api_key_env="ALT_OPENAI_API_KEY",
        api_key="explicit-key",
        base_url="http://localhost:9000/v1",
        require_api_key=False,
    )
    llm_client = BaseLLMClient()
    response = llm_client.chat(
        messages=[LLMMessage(role="user", content="hello")],
        model="gpt-default-route",
        params=LLMChatParams(),
    )
    assert response.provider == "openai"
    assert response.text == "openai:user: hello"

    # Reset to project defaults and active llama backend for deterministic follow-up tests.
    configure_openai(
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        api_key=None,
        base_url=None,
        require_api_key=True,
    )
    configure_llama_cpp_server(model="/tmp/reset-default-backend.gguf")
    shutdown_llama_cpp_server()


def test_base_llm_client_maps_openai_auth_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm_client = BaseLLMClient(backend="openai")
    with pytest.raises(LLMAuthError):
        llm_client.chat(
            messages=[LLMMessage(role="user", content="hello")],
            model="gpt-4o-mini",
            params=LLMChatParams(),
        )


def test_base_llm_client_requires_configured_llama_backend() -> None:
    shutdown_llama_cpp_server()
    llm_client = BaseLLMClient(backend="llama-cpp-server")
    with pytest.raises(LLMInvalidRequestError):
        llm_client.chat(
            messages=[LLMMessage(role="user", content="hello")],
            model="echo-test-model",
            params=LLMChatParams(),
        )
