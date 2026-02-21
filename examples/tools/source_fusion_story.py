"""Run traced source-fusion runtime example across core/script/MCP tool sources.

Expected observations:
- output includes script/core/MCP metrics over the same input text.
- report artifact path is written under ``artifacts/examples``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping

from design_research_agents import McpServer, ScriptTool, Toolbox
from design_research_agents.shared.example_support import (
    print_json,
    run_traced_callable,
    trace_info,
)


def _invoke_dict(
    runtime: Toolbox,
    tool_name: str,
    tool_input_payload: Mapping[str, object],
) -> dict[str, object]:
    tool_result = runtime.invoke(
        tool_name,
        tool_input_payload,
        request_id="example-source-fusion",
        dependencies={},
    )
    if not tool_result.ok:
        message = (
            tool_result.error.message if tool_result.error is not None else "unknown tool error"
        )
        raise RuntimeError(f"{tool_name} failed: {message}")
    if not isinstance(tool_result.result, dict):
        raise RuntimeError(f"{tool_name} returned non-object payload.")
    return tool_result.result


def _source_tool_counts(runtime: Toolbox) -> dict[str, int]:
    counts = {"core": 0, "script": 0, "mcp": 0}
    for spec in runtime.list_tools():
        if spec.name.startswith("script::"):
            counts["script"] += 1
        elif spec.name.startswith("local_core::"):
            counts["mcp"] += 1
        else:
            counts["core"] += 1
    return counts


def _run_report() -> dict[str, object]:
    runtime = Toolbox(
        workspace_root=".",
        enable_core_tools=True,
        script_tools=(
            ScriptTool(
                name="rubric_score",
                path="examples/tools/script_tools/python/rubric_score.py",
                description="Score text against a simple rubric.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "max_score": {"type": "integer"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                output_schema={"type": "object"},
                filesystem_write=True,
            ),
        ),
        mcp_servers=(
            McpServer(
                id="local_core",
                command=(sys.executable, "-m", "design_research_agents.mcp_server"),
                env={"PYTHONPATH": "src"},
                timeout_s=20,
            ),
        ),
    )

    source_text = (
        "Design review checklist: verify latch durability, reduce assembly time, "
        "and keep maintenance steps field-serviceable."
    )

    try:
        source_tool_counts = _source_tool_counts(runtime)
        write_result = _invoke_dict(
            runtime,
            "fs.write_text",
            {
                "path": "artifacts/examples/source_fusion_story_input.txt",
                "content": source_text,
                "overwrite": True,
            },
        )
        script_score = _invoke_dict(
            runtime,
            "script::rubric_score",
            {"text": source_text, "max_score": 20},
        )
        core_stats = _invoke_dict(runtime, "text.word_count", {"text": source_text})
        mcp_stats = _invoke_dict(runtime, "local_core::text.word_count", {"text": source_text})
        score_percent = _invoke_dict(
            runtime,
            "local_core::calculator",
            {"expression": (f"({script_score['score']} / {script_score['max_score']}) * 100")},
        )

        report = {
            "input_path": write_result["path"],
            "source_tool_counts": source_tool_counts,
            "script_score": script_score["score"],
            "script_max_score": script_score["max_score"],
            "core_word_count": core_stats["word_count"],
            "mcp_word_count": mcp_stats["word_count"],
            "word_count_match": core_stats["word_count"] == mcp_stats["word_count"],
            "score_percent": score_percent["result"],
            "script_trace_path": script_score.get("trace_path"),
        }
        report_write = _invoke_dict(
            runtime,
            "fs.write_text",
            {
                "path": "artifacts/examples/source_fusion_story_report.json",
                "content": json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                "overwrite": True,
            },
        )
        report["report_path"] = report_write["path"]
    finally:
        runtime.close()

    return report


def main() -> None:
    """Run traced multi-source report generation."""
    request_id = "example-tools-source-fusion-design-001"
    report = run_traced_callable(
        agent_name="ExamplesSourceFusion",
        request_id=request_id,
        input_payload={"scenario": "source-fusion-design"},
        function=_run_report,
    )
    assert isinstance(report, dict)
    report["example"] = "tools/source_fusion_story.py"
    report["trace"] = trace_info(request_id)
    print_json(report)


if __name__ == "__main__":
    main()
