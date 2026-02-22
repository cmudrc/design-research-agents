from __future__ import annotations

import importlib
import sys

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
