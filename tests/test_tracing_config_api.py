from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import design_research_agents as dra
from design_research_agents._tracing import Tracer
from design_research_agents._tracing._context import finish_trace_run, start_trace_run


def test_start_trace_run_without_tracer_is_disabled() -> None:
    scope = start_trace_run(
        agent_name="TraceConfigTestAgent",
        request_id="trace-config-test",
        input_payload={"prompt": "hello"},
        dependencies={},
        tracer=None,
    )
    assert scope is None


def test_injected_tracer_drives_trace_sink(tmp_path) -> None:
    tracer = Tracer(
        enabled=True,
        trace_dir=tmp_path / "custom-traces",
        enable_jsonl=True,
        enable_console=False,
    )

    scope = start_trace_run(
        agent_name="TraceConfigTestAgent",
        request_id="trace-config-test",
        input_payload={"prompt": "hello"},
        dependencies={},
        tracer=tracer,
    )
    finish_trace_run(scope, result=None)

    trace_files = list((tmp_path / "custom-traces").glob("run_*.jsonl"))
    assert trace_files, "Expected JSONL trace file in configured trace_dir."


def test_injected_tracer_redacts_root_run_payloads(tmp_path: Path) -> None:
    tracer = Tracer(
        enabled=True,
        trace_dir=tmp_path / "redacted-traces",
        enable_jsonl=True,
        enable_console=False,
    )

    scope = start_trace_run(
        agent_name="TraceConfigTestAgent",
        request_id="trace-redaction-test",
        input_payload={"prompt": "hello", "api_key": "secret"},
        dependencies={},
        tracer=tracer,
    )
    finish_trace_run(scope, result={"ok": True, "secret": "hidden"})

    trace_files = list((tmp_path / "redacted-traces").glob("run_*.jsonl"))
    assert trace_files
    events = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines()]
    started = next(event for event in events if event["event_type"] == "RunStarted")
    finished = next(event for event in events if event["event_type"] == "RunFinished")
    assert started["attributes"]["input"]["api_key"] == "***REDACTED***"
    assert finished["attributes"]["result"]["secret"] == "***REDACTED***"


def test_disabled_tracer_skips_trace_creation(tmp_path) -> None:
    tracer = Tracer(
        enabled=False,
        trace_dir=tmp_path / "disabled-traces",
        enable_jsonl=True,
        enable_console=True,
        console_stream=sys.stderr,
    )
    scope = start_trace_run(
        agent_name="TraceConfigTestAgent",
        request_id="trace-disabled-test",
        input_payload={"prompt": "hello"},
        dependencies={},
        tracer=tracer,
    )
    assert scope is None
    assert not (tmp_path / "disabled-traces").exists()


def test_traceconfig_not_publicly_exported() -> None:
    tracing_module = importlib.import_module("design_research_agents._tracing")
    assert "TraceConfig" not in dra.__all__
    assert not hasattr(dra, "TraceConfig")
    assert not hasattr(tracing_module, "TraceConfig")


def test_tracer_run_callable_and_trace_info(tmp_path: Path) -> None:
    tracer = Tracer(
        enabled=True,
        trace_dir=tmp_path / "trace-run-callable",
        enable_jsonl=True,
        enable_console=False,
    )
    request_id = "trace-run-callable-request"
    result = tracer.run_callable(
        agent_name="TraceRunCallableAgent",
        request_id=request_id,
        input_payload={"mode": "test"},
        function=lambda: {"ok": True, "value": 7},
    )
    assert result == {"ok": True, "value": 7}

    trace_path = tracer.resolve_latest_trace_path(request_id)
    assert isinstance(trace_path, str)
    assert trace_path.endswith(".jsonl")

    trace_info = tracer.trace_info(request_id)
    assert trace_info["request_id"] == request_id
    assert trace_info["trace_dir"] == str(tmp_path / "trace-run-callable")
    assert trace_info["trace_path"] == trace_path
