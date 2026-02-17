"""Shared deterministic stubs for workflow/runtime test modules."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence

from design_research_agents.contracts.agent import Agent, AgentResult
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.contracts.workflow import WorkflowResult, WorkflowStepResult


class SequenceLLMClient:
    """Deterministic LLM stub that returns configured responses in order."""

    def __init__(self, *, response_texts: Sequence[str]) -> None:
        self._responses = list(response_texts)

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        if not self._responses:
            raise AssertionError("No more stubbed responses available.")
        return LLMResponse(
            model=model,
            text=self._responses.pop(0),
            provider="test-sequence",
            latency_ms=4,
        )

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.chat(
            list(request.messages),
            model=request.model or self.default_model(),
            params=LLMChatParams(),
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        return "test-model"


class NoopLLMClient:
    """LLM stub used when tests inject concrete agents directly."""

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        return LLMResponse(model=model, text="{}", provider="noop")

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.chat(
            request.messages,
            model=request.model or self.default_model(),
            params=LLMChatParams(),
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        return "noop-model"


class StaticMarkerAgent(Agent):
    """Deterministic agent that always emits one marker value."""

    def __init__(self, *, marker: str) -> None:
        self._marker = marker

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del prompt, request_id, dependencies
        return AgentResult(
            output={"agent_marker": self._marker},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"agent": self._marker},
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[object]:
        del prompt, request_id, dependencies
        raise NotImplementedError


class StaticJsonDraftAgent(Agent):
    """Agent stub that always returns one JSON object in ``output.model_text``."""

    def __init__(self, *, payload: Mapping[str, object]) -> None:
        self._payload = dict(payload)
        self.run_count = 0

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del prompt, request_id, dependencies
        self.run_count += 1
        return AgentResult(
            output={"model_text": json.dumps(self._payload, ensure_ascii=True)},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"agent": "static-json-draft"},
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[object]:
        del prompt, request_id, dependencies
        raise NotImplementedError


class CaptureDependenciesAgent(Agent):
    """Agent stub that captures invocation dependencies for assertions."""

    def __init__(self) -> None:
        self.last_dependencies: Mapping[str, object] | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del prompt, request_id
        self.last_dependencies = dict(dependencies or {})
        return AgentResult(
            output={"model_text": "{}"},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"agent": "capture-deps"},
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[object]:
        del prompt, request_id, dependencies
        raise NotImplementedError


class StubToolRuntime(ToolRuntime):
    """Small in-memory tool runtime for workflow tests."""

    def __init__(
        self,
        *,
        handlers: Mapping[str, Callable[[Mapping[str, object]], Mapping[str, object]]],
    ) -> None:
        self._handlers = dict(handlers)
        self._specs = {
            name: ToolSpec(
                name=name,
                description=f"Test tool '{name}'.",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "additionalProperties": True},
            )
            for name in handlers
        }

    def list_tools(self) -> Sequence[ToolSpec]:
        return tuple(self._specs.values())

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                result={},
                error=f"Tool '{tool_name}' is not registered.",
            )

        try:
            output = dict(handler(input_dict))
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                result={},
                error=str(exc),
            )

        return ToolResult(tool_name=tool_name, ok=True, result=output)


class StaticWorkflowDelegateRunner:
    """Deterministic nested workflow used for workflow delegation tests."""

    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: str = "dag",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        del execution_mode, failure_policy, request_id, dependencies
        prompt = str((context or {}).get("prompt", ""))
        nested_step = WorkflowStepResult(
            step_id="nested_logic",
            status="completed",
            success=True,
            output={"prompt_echo": prompt},
            metadata={"stage": "execution"},
        )
        return WorkflowResult(
            success=True,
            step_results={"nested_logic": nested_step},
            execution_order=["nested_logic"],
            metadata={"runtime": "nested_stub"},
        )
