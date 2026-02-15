from __future__ import annotations

import io
import json
from pathlib import Path

from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.tracing import emitters, sinks


class _FakeSession:
    def __init__(self) -> None:
        self.root_span_id = "root-span"
        self.start_calls: list[tuple[str, str | None, dict[str, object]]] = []
        self.finish_calls: list[tuple[str, str, dict[str, object]]] = []
        self.event_calls: list[tuple[str, str, dict[str, object]]] = []

    def start_span(
        self,
        event_type: str,
        *,
        parent_span_id: str | None,
        attributes: dict[str, object],
    ) -> str:
        self.start_calls.append((event_type, parent_span_id, attributes))
        return f"span-{len(self.start_calls)}"

    def finish_span(self, event_type: str, *, span_id: str, attributes: dict[str, object]) -> None:
        self.finish_calls.append((event_type, span_id, attributes))

    def emit_span_event(
        self,
        event_type: str,
        *,
        span_id: str,
        attributes: dict[str, object],
    ) -> None:
        self.event_calls.append((event_type, span_id, attributes))


def test_jsonl_sink_appends_events(tmp_path: Path) -> None:
    out = tmp_path / "traces" / "run.jsonl"
    sink = sinks.JSONLTraceSink(out)
    sink.emit({"event_type": "A", "attributes": {"x": 1}})
    sink.emit({"event_type": "B", "attributes": {"y": 2}})
    sink.close()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "A"
    assert json.loads(lines[1])["attributes"]["y"] == 2


def test_console_sink_streaming_and_event_formatting() -> None:
    stream = io.StringIO()
    sink = sinks.ConsoleTraceSink(stream=stream)
    sink.emit({"event_type": "ModelCallToken", "attributes": {"delta_text": "abc"}})
    sink.emit({"event_type": "ModelCallFinished", "attributes": {"model": "m"}})
    sink.emit({"event_type": "RunStarted", "run_id": "r1", "attributes": {"agent": "A"}})
    sink.emit({"event_type": "ToolCallFailed", "attributes": {"tool_name": "t", "error": "boom"}})
    sink.emit({"event_type": "UnknownType", "attributes": []})
    sink.close()

    output = stream.getvalue()
    assert "abc\n[model] finished" in output
    assert "[run] start run_id=r1" in output
    assert "[tool] failed tool=t" in output


def test_console_sink_close_flushes_pending_token_line() -> None:
    stream = io.StringIO()
    sink = sinks.ConsoleTraceSink(stream=stream)
    sink.emit({"event_type": "ModelCallToken", "attributes": {"delta_text": "chunk"}})
    sink.close()
    assert stream.getvalue().endswith("\n")


def test_emitters_noop_when_tracing_disabled(monkeypatch) -> None:
    monkeypatch.setattr(emitters, "current_trace_session", lambda: None)
    assert emitters.start_model_call(model="m", messages=[], params={}) is None
    emitters.finish_model_call("span", response=None)
    emitters.emit_model_token("span", delta_text="x")


def test_model_tool_and_decision_emitters(monkeypatch) -> None:
    session = _FakeSession()
    monkeypatch.setattr(emitters, "current_trace_session", lambda: session)
    monkeypatch.setattr(emitters, "current_span_id", lambda: "current-span")

    span_id = emitters.start_model_call(
        model="gpt-test",
        messages=[{"role": "user"}],
        params={"temperature": 0.1},
        metadata={"extra": "yes"},
    )
    assert span_id == "span-1"
    assert session.start_calls[0][0] == "ModelCallStarted"
    assert session.start_calls[0][1] == "current-span"
    assert session.start_calls[0][2]["extra"] == "yes"

    emitters.finish_model_call(
        span_id,
        error="failed",
        response=LLMResponse(text="t", model="m1"),
    )
    assert session.finish_calls[-1][0] == "ModelCallFailed"
    assert session.finish_calls[-1][2]["model"] == "m1"

    emitters.finish_model_call(span_id, response=LLMResponse(text="ok", model="m2"))
    assert session.finish_calls[-1][0] == "ModelCallFinished"
    assert session.finish_calls[-1][2]["model"] == "m2"

    emitters.emit_model_token(span_id, delta_text="")
    assert not [x for x in session.event_calls if x[0] == "ModelCallToken"]
    emitters.emit_model_token(span_id, delta_text="abc")
    assert session.event_calls[-1][0] == "ModelCallToken"

    tool_span = emitters.start_tool_call(
        tool_name="calculator",
        tool_input={"expression": "1+1"},
        request_id="r1",
        dependencies={"z": 1, "a": 2},
    )
    assert tool_span == "span-2"
    assert session.start_calls[-1][0] == "ToolCallStarted"
    assert session.start_calls[-1][2]["dependency_keys"] == ["a", "z"]

    emitters.finish_tool_call(tool_span, tool_name="calculator", result={"result": 2})
    assert session.finish_calls[-1][0] == "ToolCallFinished"
    emitters.fail_tool_call(tool_span, tool_name="calculator", error="boom")
    assert session.finish_calls[-1][0] == "ToolCallFailed"

    emitters.emit_router_decision(
        source="fallback",
        alternatives=["a", "b"],
        selected_tool_name="a",
        selected_index=0,
        reason="test",
        parsed_route={"selected": "a"},
    )
    emitters.emit_model_selection_decision(
        model_id="m",
        provider="p",
        rationale="r",
        policy_id="pid",
        policy_config={},
        catalog_signature="sig",
        intent={},
        constraints={},
        hardware_profile={},
        candidate_count=3,
    )
    emitters.emit_tool_selection_decision(
        source="model",
        tool_name="calculator",
        reason="good",
        parsed_tool_call={"tool_name": "calculator"},
    )
    emitters.emit_continuation_decision(step=1, should_continue=False, reason="stop", source="x")
    emitters.emit_guardrail_decision(
        guardrail="schema",
        decision="deny",
        reason="invalid",
        details={"field": "x"},
    )

    event_types = [item[0] for item in session.event_calls]
    assert "RouterDecision" in event_types
    assert "ModelSelectionDecision" in event_types
    assert "ToolSelectionDecision" in event_types
    assert "ContinuationDecision" in event_types
    assert "GuardrailDecision" in event_types
