"""Pure workflow runtime example with chained tool and logic steps."""

from __future__ import annotations

import design_research_agents as dra


def main() -> None:
    """Run the pure tool workflow example and print serialized output."""
    runtime = dra.workflows.WorkflowRuntime(
        tool_runtime=dra.tools.UnifiedToolRuntime(),
    )
    steps = [
        dra.workflows.ToolStep(
            step_id="seed_csv",
            tool_name="fs.write_text",
            input_data={
                "path": "artifacts/examples/workflow_tool_inventory.csv",
                "content": (
                    "tool,source\n"
                    "calculator,core\n"
                    "search.ripgrep,core\n"
                    "repo_quickscan,lazy\n"
                    "local_core::calculator,mcp\n"
                ),
                "overwrite": True,
            },
        ),
        dra.workflows.ToolStep(
            step_id="describe_csv",
            tool_name="data.describe",
            dependencies=("seed_csv",),
            input_builder=lambda context: {
                "path": context["dependency_results"]["seed_csv"]["output"]["result"]["path"],
                "kind": "csv",
            },
        ),
        dra.workflows.ToolStep(
            step_id="scan_sources",
            tool_name="search.ripgrep",
            dependencies=("describe_csv",),
            input_builder=lambda context: {
                "query": "UnifiedToolRuntime",
                "root": "src/design_research_agents/tools",
                "max_matches": 6,
            },
        ),
        dra.workflows.LogicStep(
            step_id="finalize",
            dependencies=("describe_csv", "scan_sources"),
            handler=lambda context: {
                "csv_rows": context["dependency_results"]["describe_csv"]["output"]["result"][
                    "rows"
                ],
                "csv_columns": context["dependency_results"]["describe_csv"]["output"]["result"][
                    "columns"
                ],
                "runtime_reference_hits": context["dependency_results"]["scan_sources"]["output"][
                    "result"
                ]["count"],
            },
        ),
    ]

    result = runtime.run(steps, execution_mode="sequential")
    print(result.asdict())


if __name__ == "__main__":
    main()
