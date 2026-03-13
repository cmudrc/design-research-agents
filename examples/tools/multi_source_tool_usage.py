"""# Tools / Multi-Source Tool Usage.

## Introduction
MCP standardizes tool connectivity, data-fusion concepts motivate combining heterogeneous signals, and RAG
provides a grounding mechanism for synthesis over retrieved evidence. This example fuses MCP tools and
script tools into one workflow that emits a traceable narrative artifact.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Toolbox.invoke_dict(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["Toolbox.invoke_dict(...)"]
    C --> D["core, script, and MCP tools execute in one composed runtime"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "core_word_count": 14,
     "example": "tools/multi_source_tool_usage.py",
     "input_path": "artifacts/examples/<truncated-input-path>",
     "mcp_word_count": 14,
     "report_path": "artifacts/examples/<truncated-report-path>",
     "score_percent": 10.0,
     "script_max_score": 20,
     "script_score": 2,
     "script_trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-script-rubric-score-001.jsonl",
     "source_tool_counts": {
       "core": 23,
       "mcp": 23,
       "script": 1
     },
     "trace": {
       "request_id": "example-tools-multi-source-tool-usage-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-tools-multi-source-tool-usage-001.jsonl"
     },
     "word_count_match": true
   }


## References
- `Model Context Protocol Specification <https://modelcontextprotocol.io/specification/2025-06-18>`_
- `Data Fusion (Wikipedia) <https://en.wikipedia.org/wiki/Data_fusion>`_
- `Retrieval-Augmented Generation <https://arxiv.org/abs/2005.11401>`_
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import design_research_agents as drag


def _source_tool_counts(runtime: drag.Toolbox) -> dict[str, int]:
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
    source_text = (
        "Design review checklist: verify latch durability, reduce assembly time, "
        "and keep maintenance steps field-serviceable."
    )

    with drag.Toolbox(
        workspace_root=".",
        enable_core_tools=True,
        script_tools=(
            drag.ScriptToolConfig(
                name="rubric_score",
                path="examples/tools/script_tools/rubric_score.py",
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
            drag.MCPServerConfig(
                id="local_core",
                command=(sys.executable, "-m", "design_research_agents._mcp_server"),
                env={"PYTHONPATH": "src"},
                timeout_s=20,
            ),
        ),
    ) as runtime:
        source_tool_counts = _source_tool_counts(runtime)
        write_result = runtime.invoke_dict(
            "fs.write_text",
            {
                "path": "artifacts/examples/multi_source_tool_usage_input.txt",
                "content": source_text,
                "overwrite": True,
            },
            request_id="example-multi-source-tool-usage",
            dependencies={},
        )
        script_score = runtime.invoke_dict(
            "script::rubric_score",
            {"text": source_text, "max_score": 20},
            request_id="example-multi-source-tool-usage",
            dependencies={},
        )
        core_stats = runtime.invoke_dict(
            "text.word_count",
            {"text": source_text},
            request_id="example-multi-source-tool-usage",
            dependencies={},
        )
        mcp_stats = runtime.invoke_dict(
            "local_core::text.word_count",
            {"text": source_text},
            request_id="example-multi-source-tool-usage",
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
                "path": "artifacts/examples/multi_source_tool_usage_report.json",
                "content": json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                "overwrite": True,
            },
            request_id="example-multi-source-tool-usage",
            dependencies={},
        )
        report["report_path"] = report_write["path"]

    return report


def main() -> None:
    """Run traced multi-source report generation."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-tools-multi-source-tool-usage-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    report = tracer.run_callable(
        agent_name="ExamplesMultiSourceToolUsage",
        request_id=request_id,
        input_payload={"scenario": "multi-source-tool-usage"},
        function=_run_report,
    )
    assert isinstance(report, dict)
    report["example"] = "tools/multi_source_tool_usage.py"
    report["trace"] = tracer.trace_info(request_id)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
