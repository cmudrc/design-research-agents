"""Source-fusion runtime example combining core, script, and MCP tools."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping

from design_research_agents import Toolbox
from design_research_agents.tools.config import McpServer, ScriptTool


def _invoke_dict(
    runtime: Toolbox,
    tool_name: str,
    tool_input_payload: Mapping[str, object],
) -> dict[str, object]:
    """Run invoke dict.

    Args:
        runtime: Parameter value.
        tool_name: Parameter value.
        tool_input_payload: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
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


def main() -> None:
    """Run a deterministic multi-source story and persist a JSON artifact."""
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

    story_text = (
        "We combined core, script, and mcp tools to create one deterministic research report "
        "for the runtime examples folder."
    )

    try:
        write_result = _invoke_dict(
            runtime,
            "fs.write_text",
            {
                "path": "artifacts/examples/source_fusion_story_text.txt",
                "content": story_text,
                "overwrite": True,
            },
        )
        script_score = _invoke_dict(
            runtime,
            "script::rubric_score",
            {"text": story_text, "max_score": 20},
        )
        mcp_stats = _invoke_dict(runtime, "local_core::text.word_count", {"text": story_text})
        source_hits = _invoke_dict(
            runtime,
            "search.ripgrep",
            {
                "query": "Toolbox",
                "root": "src/design_research_agents/tools",
                "max_matches": 5,
            },
        )
        combined = _invoke_dict(
            runtime,
            "local_core::calculator",
            {"expression": f"{script_score['score']} + {mcp_stats['word_count']}"},
        )

        report = {
            "story_path": write_result["path"],
            "script_score": script_score["score"],
            "script_max_score": script_score["max_score"],
            "mcp_word_count": mcp_stats["word_count"],
            "source_hit_count": source_hits["count"],
            "combined_metric": combined["result"],
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

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
