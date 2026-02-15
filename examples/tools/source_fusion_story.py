"""Source-fusion runtime example combining core, lazy, and MCP tools."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping

import design_research_agents as dra


def _invoke_dict(
    runtime: dra.tools.UnifiedToolRuntime,
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


def main() -> None:
    """Run a deterministic multi-source story and persist a JSON artifact."""
    runtime = dra.tools.UnifiedToolRuntime(
        config=dra.tools.ToolRuntimeConfig(
            core_tools=dra.tools.CoreToolsConfig(workspace_root="."),
            lazy_tools=dra.tools.LazyToolsConfig(
                enabled=True, search_paths=("examples/lazy_tools",)
            ),
            mcp=dra.tools.McpConfig(
                enabled=True,
                servers=(
                    dra.tools.McpServerConfig(
                        id="local_core",
                        command=(sys.executable, "-m", "design_research_agents.mcp_server"),
                        env={"PYTHONPATH": "src"},
                        timeout_s=20,
                    ),
                ),
            ),
        )
    )

    story_text = (
        "We combined core, lazy, and mcp tools to create one deterministic research report "
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
        lazy_score = _invoke_dict(
            runtime,
            "lazy::rubric_score",
            {"text": story_text, "max_score": 20},
        )
        mcp_stats = _invoke_dict(runtime, "local_core::text.word_count", {"text": story_text})
        source_hits = _invoke_dict(
            runtime,
            "search.ripgrep",
            {
                "query": "UnifiedToolRuntime",
                "root": "src/design_research_agents/tools",
                "max_matches": 5,
            },
        )
        combined = _invoke_dict(
            runtime,
            "local_core::calculator",
            {"expression": f"{lazy_score['score']} + {mcp_stats['word_count']}"},
        )

        report = {
            "story_path": write_result["path"],
            "lazy_score": lazy_score["score"],
            "lazy_max_score": lazy_score["max_score"],
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
