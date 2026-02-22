from __future__ import annotations

import json
from pathlib import Path

from design_research_agents._tracing import _analysis as analysis
from design_research_agents._tracing._analysis import analyze_trace_dir, main


def _write_trace(path: Path, events: list[dict[str, object]], *, malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted keys keep fixture traces stable for readable diffs.
    lines = [json.dumps(event, sort_keys=True) for event in events]
    if malformed:
        lines.append("{not-json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analyze_trace_dir_aggregates_extended_metrics(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    _write_trace(
        trace_dir / "run_001_alpha.jsonl",
        [
            {"event_type": "RunFinished", "run_id": "r1", "attributes": {"success": True}},
            {
                "event_type": "ModelRequestObserved",
                "run_id": "r1",
                "attributes": {"model": "gpt-a", "request": {"model": "gpt-a"}},
            },
            {
                "event_type": "ModelResponseObserved",
                "run_id": "r1",
                "attributes": {
                    "model": "gpt-a",
                    "response": {
                        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
                        "latency_ms": 130,
                    },
                    "error": None,
                },
            },
            {
                "event_type": "ToolInvocationObserved",
                "run_id": "r1",
                "attributes": {"tool_name": "text.word_count"},
            },
            {
                "event_type": "ToolResultObserved",
                "run_id": "r1",
                "attributes": {"tool_name": "text.word_count", "ok": True},
            },
            {
                "event_type": "WorkflowStepResultObserved",
                "run_id": "r1",
                "attributes": {"step_type": "logic", "status": "completed"},
            },
            {
                "event_type": "ToolCallFinished",
                "run_id": "r1",
                "duration_ms": 33,
                "attributes": {"tool_name": "text.word_count"},
            },
        ],
        malformed=True,
    )
    _write_trace(
        trace_dir / "run_002_beta.jsonl",
        [
            {"event_type": "RunFinished", "run_id": "r2", "attributes": {"success": False}},
            {
                "event_type": "ModelRequestObserved",
                "run_id": "r2",
                "attributes": {"model": "gpt-b", "request": {"model": "gpt-b"}},
            },
            {
                "event_type": "ModelResponseObserved",
                "run_id": "r2",
                "attributes": {"model": "gpt-b", "response": {}, "error": "provider failure"},
            },
            {
                "event_type": "ToolInvocationObserved",
                "run_id": "r2",
                "attributes": {"tool_name": "text.word_count"},
            },
            {
                "event_type": "ToolResultObserved",
                "run_id": "r2",
                "attributes": {"tool_name": "text.word_count", "ok": False, "error": "tool failed"},
            },
            {
                "event_type": "WorkflowStepResultObserved",
                "run_id": "r2",
                "attributes": {"step_type": "tool", "status": "failed"},
            },
        ],
    )

    summary = analyze_trace_dir(trace_dir=trace_dir, glob_pattern="run_*.jsonl", top_n=5)
    # Assert cross-section metrics so regressions in one aggregator path are caught quickly.
    assert summary["runs"]["unique_run_count"] == 2
    assert summary["runs"]["success_count"] == 1
    assert summary["runs"]["failure_count"] == 1
    assert summary["events"]["malformed_lines"] == 1
    assert summary["models"]["model_call_count"] == 2
    assert summary["models"]["tokens"]["total_tokens"] == 16
    assert summary["models"]["model_failure_count"] == 1
    assert summary["tools"]["tool_invocation_count"] == 2
    assert summary["tools"]["tool_failure_count"] == 1
    assert summary["workflow_steps"]["step_execution_count"] == 2
    assert summary["workflow_steps"]["counts_by_status"]["completed"] == 1
    assert summary["workflow_steps"]["counts_by_status"]["failed"] == 1
    assert summary["errors"]["error_event_count"] >= 2


def test_analyze_trace_dir_falls_back_to_legacy_event_types(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    _write_trace(
        trace_dir / "run_legacy.jsonl",
        [
            {"event_type": "RunFinished", "run_id": "legacy", "attributes": {"success": True}},
            {"event_type": "ModelCallStarted", "run_id": "legacy", "attributes": {"model": "legacy-model"}},
            {
                "event_type": "ModelCallFinished",
                "run_id": "legacy",
                "attributes": {"model": "legacy-model", "response": {"usage": {"total_tokens": 4}}},
            },
            {"event_type": "ToolCallStarted", "run_id": "legacy", "attributes": {"tool_name": "legacy-tool"}},
            {
                "event_type": "ToolCallFailed",
                "run_id": "legacy",
                "attributes": {"tool_name": "legacy-tool", "error": "boom"},
            },
            {
                "event_type": "WorkflowStepFinished",
                "run_id": "legacy",
                "attributes": {"step_type": "tool", "status": "failed"},
            },
        ],
    )

    summary = analyze_trace_dir(trace_dir=trace_dir, glob_pattern="run_*.jsonl", top_n=5)
    # Legacy event compatibility is critical for old trace archives.
    assert summary["models"]["model_call_count"] == 1
    assert summary["models"]["tokens"]["total_tokens"] == 4
    assert summary["tools"]["tool_invocation_count"] == 1
    assert summary["tools"]["tool_failure_count"] == 1
    assert summary["workflow_steps"]["counts_by_status"]["failed"] == 1


def test_analysis_cli_json_output_shape(tmp_path: Path, capsys) -> None:
    trace_dir = tmp_path / "traces"
    _write_trace(
        trace_dir / "run_cli.jsonl", [{"event_type": "RunFinished", "run_id": "cli", "attributes": {"success": True}}]
    )

    exit_code = main(["--trace-dir", str(trace_dir), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert sorted(payload.keys()) == [
        "errors",
        "events",
        "inputs",
        "latency",
        "models",
        "runs",
        "tools",
        "warnings",
        "workflow_steps",
    ]


def test_analysis_internal_helpers_cover_fallback_and_edge_paths(tmp_path: Path) -> None:
    trace_file = tmp_path / "traces" / "run_helpers.jsonl"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(
        "\n".join(
            [
                "",
                json.dumps({"event_type": "AgentRunFinished", "run_id": "r1", "attributes": {"success": True}}),
                json.dumps({"event_type": "RunFinished", "run_id": "r2", "attributes": {"success": False}}),
                json.dumps({"event_type": "ModelCallStarted", "run_id": "r2", "attributes": {"model": "legacy"}}),
                json.dumps(
                    {
                        "event_type": "ModelCallFinished",
                        "run_id": "r2",
                        "duration_ms": 44,
                        "attributes": {
                            "response": {"model": "legacy", "usage": {"prompt_tokens": 1.0, "total_tokens": True}}
                        },
                    }
                ),
                json.dumps(
                    {"event_type": "ToolCallStarted", "run_id": "r2", "attributes": {"tool_name": "legacy-tool"}}
                ),
                json.dumps({"event_type": "ToolCallFinished", "run_id": "r2", "duration_ms": 12, "attributes": {}}),
                json.dumps(
                    {
                        "event_type": "ToolCallFailed",
                        "run_id": "r2",
                        "duration_ms": 13,
                        "attributes": {"error": "boom"},
                    }
                ),
                json.dumps({"event_type": "WorkflowStepFinished", "run_id": "r2", "attributes": {"status": "failed"}}),
                json.dumps({"event_type": "UnknownType", "run_id": "", "attributes": []}),
                "{not-json",
                "[]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events, malformed = analysis._read_trace_file(trace_file)
    assert malformed == 2
    assert len(events) >= 8

    summary = analyze_trace_dir(trace_dir=trace_file.parent, glob_pattern="run_helpers.jsonl", top_n=3)
    assert summary["runs"]["runs_with_terminal_status"] == 2
    assert summary["models"]["model_call_count"] == 1
    assert summary["tools"]["tool_invocation_count"] == 1
    assert summary["workflow_steps"]["step_execution_count"] == 1
    assert summary["latency"]["tool_duration_ms"]["count"] == 2

    model_name_from_response = analysis._resolve_model_name({"attributes": {"response": {"model": "from-response"}}})
    model_name_from_request = analysis._resolve_model_name({"attributes": {"request": {"model": "from-request"}}})
    model_name_unknown = analysis._resolve_model_name({"attributes": {}})
    assert model_name_from_response == "from-response"
    assert model_name_from_request == "from-request"
    assert model_name_unknown == "unknown"

    assert analysis._resolve_tool_name({"attributes": {}}) == "unknown"
    assert analysis._extract_usage({"attributes": {"usage": {"prompt_tokens": 1}}}) == {"prompt_tokens": 1}
    assert analysis._extract_usage({"attributes": {"response": {"usage": {"completion_tokens": 2}}}}) == {
        "completion_tokens": 2
    }
    assert analysis._extract_response_latency({"attributes": {"response": {"latency_ms": 9}}}) == 9
    assert analysis._extract_response_latency({"duration_ms": 5, "attributes": {}}) == 5
    assert analysis._extract_tool_success({"event_type": "ToolCallFinished", "attributes": {}}) is True
    assert analysis._extract_tool_success({"event_type": "ToolCallFailed", "attributes": {}}) is False
    assert analysis._extract_tool_success({"event_type": "Other", "attributes": {}}) is None
    assert analysis._extract_tool_success({"event_type": "ToolResultObserved", "attributes": {"ok": True}}) is True
    assert analysis._extract_tool_success({"event_type": "ToolResultObserved", "attributes": {"error": "x"}}) is False
    assert analysis._extract_error_text({"attributes": {"error": {"message": "mapped"}}}) == "mapped"
    assert analysis._extract_error_text({"attributes": {"error": {"code": 1}}}) == '{"code": 1}'
    assert analysis._extract_error_text({"attributes": {"error": ["a"]}}) == '["a"]'

    assert analysis._summarize_numbers([])["count"] == 0
    assert analysis._percentile([], 0.95) == 0
    assert analysis._coerce_int(True) is None
    assert analysis._coerce_int(1.2) == 1
    assert analysis._coerce_bool(1) is None
    assert analysis._as_mapping("x") == {}
    assert analysis._mapping_value({"a": []}, "a") == {}
    assert analysis.cast_mapping({1: "x"}) == {"1": "x"}
    assert analysis._format_human_summary(summary).startswith("Trace Analysis Summary")


def test_analysis_cli_human_and_broken_pipe_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    trace_dir = tmp_path / "traces"
    _write_trace(
        trace_dir / "run_human.jsonl",
        [{"event_type": "RunFinished", "run_id": "cli", "attributes": {"success": True}}],
    )

    exit_code = main(["--trace-dir", str(trace_dir)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Trace Analysis Summary" in captured.out

    monkeypatch.setattr("builtins.print", lambda *_args, **_kwargs: (_ for _ in ()).throw(BrokenPipeError()))
    assert main(["--trace-dir", str(trace_dir), "--json"]) == 0
