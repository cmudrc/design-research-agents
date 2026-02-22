"""Example script.

Motivation
Run traced source-fusion runtime example across core/script/MCP tool sources.

Diagram
```mermaid
flowchart LR
    A["Tool input"] --> B["Tool runtime"]
    B --> C["source fusion story result"]
    C --> D["Artifacts and trace"]
```

Technical Walkthrough
1. Configure the runtime surface for `tools` use-cases and run `source_fusion_story`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `DRA_EXAMPLE_MCP_COMMAND='python3 -m your_mcp_server_module'`
`PYTHONPATH=src python3 examples/tools/source_fusion_story.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from design_research_agents import McpServer, ScriptTool, Toolbox, Tracer


def _mcp_server_command() -> tuple[str, ...]:
    raw_command = os.environ.get("DRA_EXAMPLE_MCP_COMMAND")
    if raw_command is None or not raw_command.strip():
        raise RuntimeError(
            "Set DRA_EXAMPLE_MCP_COMMAND to a stdio MCP server command "
            "(for example: 'python3 -m your_mcp_server_module')."
        )
    return tuple(shlex.split(raw_command))


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
                command=_mcp_server_command(),
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
        write_result = runtime.invoke_dict(
            "fs.write_text",
            {
                "path": "artifacts/examples/source_fusion_story_input.txt",
                "content": source_text,
                "overwrite": True,
            },
            request_id="example-source-fusion",
            dependencies={},
        )
        script_score = runtime.invoke_dict(
            "script::rubric_score",
            {"text": source_text, "max_score": 20},
            request_id="example-source-fusion",
            dependencies={},
        )
        core_stats = runtime.invoke_dict(
            "text.word_count",
            {"text": source_text},
            request_id="example-source-fusion",
            dependencies={},
        )
        mcp_stats = runtime.invoke_dict(
            "local_core::text.word_count",
            {"text": source_text},
            request_id="example-source-fusion",
            dependencies={},
        )
        score_percent = (float(script_score["score"]) / float(script_score["max_score"])) * 100.0

        report = {
            "input_path": write_result["path"],
            "source_tool_counts": source_tool_counts,
            "script_score": script_score["score"],
            "script_max_score": script_score["max_score"],
            "core_word_count": core_stats["word_count"],
            "mcp_word_count": mcp_stats["word_count"],
            "word_count_match": core_stats["word_count"] == mcp_stats["word_count"],
            "score_percent": score_percent,
            "script_trace_path": script_score.get("trace_path"),
        }
        report_write = runtime.invoke_dict(
            "fs.write_text",
            {
                "path": "artifacts/examples/source_fusion_story_report.json",
                "content": json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                "overwrite": True,
            },
            request_id="example-source-fusion",
            dependencies={},
        )
        report["report_path"] = report_write["path"]
    finally:
        runtime.close()

    return report


def main() -> None:
    """Run traced multi-source report generation."""
    request_id = "example-tools-source-fusion-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    report = tracer.run_callable(
        agent_name="ExamplesSourceFusion",
        request_id=request_id,
        input_payload={"scenario": "source-fusion-design"},
        function=_run_report,
    )
    assert isinstance(report, dict)
    report["example"] = "tools/source_fusion_story.py"
    report["trace"] = tracer.trace_info(request_id)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
