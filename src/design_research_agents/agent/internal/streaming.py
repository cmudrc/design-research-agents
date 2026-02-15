"""Streaming helpers for accumulating LLM deltas into responses."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from design_research_agents.contracts.llm import (
    LLMDelta,
    LLMResponse,
    ToolCall,
    ToolCallDelta,
    Usage,
)


@dataclass(slots=True)
class StreamAccumulator:
    """Collect streaming deltas into a response payload."""

    text_parts: list[str] = field(default_factory=list)
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage: Usage | None = None

    def apply(self, delta: LLMDelta) -> None:
        if delta.text_delta:
            self.text_parts.append(delta.text_delta)
        if delta.tool_call_delta:
            self._apply_tool_call_delta(delta.tool_call_delta)
        if delta.usage_delta:
            self.usage = delta.usage_delta

    def text(self) -> str:
        return "".join(self.text_parts)

    def _apply_tool_call_delta(self, delta: ToolCallDelta) -> None:
        call_id = delta.call_id or "call_1"
        entry = self.tool_calls.setdefault(
            call_id,
            {
                "name": delta.name,
                "arguments": "",
            },
        )
        if delta.name:
            entry["name"] = delta.name
        if delta.arguments_json_delta:
            entry["arguments"] += delta.arguments_json_delta

    def build_tool_calls(self) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for call_id, payload in self.tool_calls.items():
            name = payload.get("name") or ""
            arguments = payload.get("arguments") or ""
            calls.append(ToolCall(name=name, arguments_json=arguments, call_id=call_id))
        return tuple(calls)


def finalize_stream_response(
    *,
    stream: Iterator[LLMDelta],
    accumulator: StreamAccumulator,
    model: str,
) -> LLMResponse:
    response = getattr(stream, "response", None)
    if isinstance(response, LLMResponse):
        if not response.text and accumulator.text_parts:
            return LLMResponse(
                text=accumulator.text(),
                tool_calls=response.tool_calls,
                usage=response.usage,
                raw=response.raw,
                provenance=response.provenance,
                model=response.model,
                provider=response.provider,
                finish_reason=response.finish_reason,
                latency_ms=response.latency_ms,
            )
        return response
    return LLMResponse(
        text=accumulator.text(),
        tool_calls=accumulator.build_tool_calls(),
        usage=accumulator.usage,
        model=model,
    )
