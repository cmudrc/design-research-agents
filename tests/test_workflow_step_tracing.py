from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMMessage, LLMRequest, LLMResponse
from design_research_agents._contracts._workflow import (
    DelegateBatchStep,
    DelegateStep,
    LogicStep,
    LoopStep,
    ModelStep,
    ToolStep,
)
from design_research_agents._runtime._workflow import _step_tracing as step_tracing
from design_research_agents._tracing import _context as tracing_context
from design_research_agents._tracing._context import current_span_id


class _NoopDelegate:
    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return ExecutionResult(success=True, output={"ok": True})


class _NoopLLMClient:
    def default_model(self) -> str:
        return "test-model"

    def close(self) -> None:
        return None

    def __enter__(self) -> _NoopLLMClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None

    def chat(self, messages: Sequence[LLMMessage], *, model: str, params: object) -> LLMResponse:
        del messages, params
        return LLMResponse(model=model, text="ok", provider="test")

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.chat(
            request.messages,
            model=request.model or self.default_model(),
            params=object(),
        )


class _RecordingSession:
    def __init__(self) -> None:
        self.root_span_id = "root-span"
        self.started: list[tuple[str, str | None, dict[str, object]]] = []
        self.finished: list[tuple[str, str, dict[str, object]]] = []

    def start_span(
        self,
        event_type: str,
        *,
        parent_span_id: str | None,
        attributes: dict[str, object],
    ) -> str:
        self.started.append((event_type, parent_span_id, dict(attributes)))
        return "step-span-id"

    def finish_span(
        self,
        event_type: str,
        *,
        span_id: str,
        attributes: dict[str, object],
    ) -> None:
        self.finished.append((event_type, span_id, dict(attributes)))


def test_step_kind_labels_all_supported_step_types() -> None:
    tool_step = ToolStep(step_id="tool", tool_name="sum")
    agent_step = DelegateStep(step_id="agent", delegate=_NoopDelegate(), prompt="go")
    model_step = ModelStep(
        step_id="model",
        llm_client=_NoopLLMClient(),
        request_builder=lambda context: LLMRequest(
            messages=[LLMMessage(role="user", content=str(context.get("prompt", "")))],
            model="test-model",
        ),
    )
    delegate_batch_step = DelegateBatchStep(step_id="batch", calls_builder=lambda context: [])
    loop_step = LoopStep(step_id="loop", steps=(tool_step,))
    logic_step = LogicStep(step_id="logic", handler=lambda context: dict(context))

    assert step_tracing.step_kind(tool_step) == "tool"
    assert step_tracing.step_kind(agent_step) == "agent"
    assert step_tracing.step_kind(model_step) == "model"
    assert step_tracing.step_kind(delegate_batch_step) == "delegate_batch"
    assert step_tracing.step_kind(loop_step) == "loop"
    assert step_tracing.step_kind(logic_step) == "logic"


def test_start_step_span_returns_none_without_active_session(monkeypatch) -> None:
    monkeypatch.setattr(step_tracing, "current_trace_session", lambda: None)

    span_id = step_tracing.start_step_span(step=ToolStep(step_id="tool", tool_name="sum"), step_id="tool")

    assert span_id is None


def test_start_step_span_prefers_active_parent_and_falls_back_to_root(monkeypatch) -> None:
    session = _RecordingSession()
    monkeypatch.setattr(step_tracing, "current_trace_session", lambda: session)
    monkeypatch.setattr(step_tracing, "current_span_id", lambda: "active-parent")

    started_span_id = step_tracing.start_step_span(
        step=ToolStep(step_id="tool", tool_name="sum"),
        step_id="tool",
    )

    assert started_span_id == "step-span-id"
    assert session.started[0][0] == "WorkflowStepStarted"
    assert session.started[0][1] == "active-parent"
    assert session.started[0][2]["step_id"] == "tool"
    assert session.started[0][2]["step_type"] == "tool"
    assert session.started[0][2]["dependencies"] == []

    monkeypatch.setattr(step_tracing, "current_span_id", lambda: None)
    step_tracing.start_step_span(
        step=ToolStep(step_id="tool-2", tool_name="sum", dependencies=("dep-a",)),
        step_id="tool-2",
    )
    assert session.started[1][1] == "root-span"
    assert session.started[1][2]["dependencies"] == ["dep-a"]


def test_finish_step_span_is_noop_without_session_or_span(monkeypatch) -> None:
    session = _RecordingSession()
    monkeypatch.setattr(step_tracing, "current_trace_session", lambda: None)
    step_tracing.finish_step_span(
        span_id="step-span-id",
        step_id="tool",
        status="success",
        error=None,
    )

    monkeypatch.setattr(step_tracing, "current_trace_session", lambda: session)
    step_tracing.finish_step_span(
        span_id=None,
        step_id="tool",
        status="success",
        error=None,
    )
    assert session.finished == []

    step_tracing.finish_step_span(
        span_id="step-span-id",
        step_id="tool",
        status="failed",
        error="boom",
    )
    assert session.finished == [
        (
            "WorkflowStepFinished",
            "step-span-id",
            {"step_id": "tool", "status": "failed", "error": "boom"},
        )
    ]


def test_activate_step_span_sets_and_restores_active_span() -> None:
    outer_token = tracing_context._CURRENT_SPAN.set("outer-span")
    try:
        with step_tracing.activate_step_span(None):
            assert current_span_id() == "outer-span"

        with step_tracing.activate_step_span("inner-span"):
            assert current_span_id() == "inner-span"

        assert current_span_id() == "outer-span"
    finally:
        tracing_context._CURRENT_SPAN.reset(outer_token)
