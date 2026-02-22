from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMCapabilityError,
    LLMDelta,
    LLMInvalidRequestError,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents._contracts._tools import ToolSpec
from design_research_agents.llm._backends._base import (
    BaseLLMBackend,
    _build_tool_call_schema,
    _matches_model_pattern,
    _merge_raw,
    _normalize_json_text,
    _parse_tool_calls,
)


class _BackendStub(BaseLLMBackend):
    def __init__(self, *, capabilities: BackendCapabilities) -> None:
        super().__init__(
            name="stub",
            kind="test",
            default_model="stub-model",
            base_url=None,
            config_hash="stub-hash",
            model_patterns=("stub-*",),
        )
        self._capabilities = capabilities
        self.generate_calls: list[LLMRequest] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def healthcheck(self) -> BackendStatus:
        return BackendStatus(ok=True, message="ok")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls.append(request)
        return LLMResponse(text="{}", model=request.model, provider="stub")

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        yield LLMDelta(text_delta=request.model)


def _request(
    *,
    model: str = "stub-model",
    tools: Sequence[ToolSpec] = (),
    response_schema: dict[str, object] | None = None,
) -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role="user", content="hello")],
        model=model,
        tools=tools,
        response_schema=response_schema,
        response_format=None,
        provider_options={},
        metadata={},
        task_profile=None,
    )


def test_base_backend_stream_and_embed_capability_guards() -> None:
    backend = _BackendStub(
        capabilities=BackendCapabilities(
            streaming=False,
            tool_calling="none",
            json_mode="none",
            vision=False,
            max_context_tokens=None,
        )
    )

    with pytest.raises(LLMCapabilityError, match="streaming"):
        list(backend.stream(_request()))
    with pytest.raises(LLMCapabilityError, match="embeddings"):
        backend.embed(["hello"])


def test_base_backend_generate_enforces_tool_and_json_capabilities() -> None:
    backend = _BackendStub(
        capabilities=BackendCapabilities(
            streaming=True,
            tool_calling="none",
            json_mode="none",
            vision=False,
            max_context_tokens=None,
        )
    )
    tool = ToolSpec(
        name="calculator",
        description="Compute",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    with pytest.raises(LLMCapabilityError, match="tool calling"):
        backend.generate(_request(tools=(tool,)))
    with pytest.raises(LLMCapabilityError, match="JSON output"):
        backend.generate(_request(response_schema={"type": "object"}))
    with pytest.raises(LLMInvalidRequestError, match="both tools and response_schema"):
        backend.generate(_request(tools=(tool,), response_schema={"type": "object"}))


def test_base_backend_helper_parsers_and_patterns() -> None:
    tools = (
        ToolSpec(
            name="alpha",
            description="Alpha",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
        ToolSpec(
            name="beta",
            description="Beta",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
    )

    schema = _build_tool_call_schema(tools)
    assert schema["required"] == ["tool_calls"]
    assert schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"] == [
        "alpha",
        "beta",
    ]

    parsed_calls = _parse_tool_calls(
        {
            "tool_calls": [
                {"name": "alpha", "arguments": {"x": 1}},
                {"name": "beta", "arguments": {}, "id": "beta-call"},
            ]
        },
        tools,
    )
    assert parsed_calls[0].call_id == "call_1"
    assert parsed_calls[1].call_id == "beta-call"

    parsed_single = _parse_tool_calls({"tool_name": "alpha", "tool_input": {"x": 2}}, tools)
    assert len(parsed_single) == 1
    assert parsed_single[0].name == "alpha"

    with pytest.raises(ValueError, match="Unknown tool name"):
        _parse_tool_calls([{"name": "missing", "arguments": {}}], tools)

    assert _matches_model_pattern("stub-model", "stub-*") is True
    assert _matches_model_pattern("stub-model", "other-*") is False
    assert _matches_model_pattern("stub-model", "stub-model") is True
    assert _matches_model_pattern("stub-model", "stub-*-v2") is False

    assert _normalize_json_text({"z": 1, "a": 2}) == '{"a": 2, "z": 1}'
    assert _merge_raw({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
