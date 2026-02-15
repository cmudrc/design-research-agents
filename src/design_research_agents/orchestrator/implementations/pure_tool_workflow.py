"""Reusable pure-tool workflow orchestration chunk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.orchestrator import (
    LogicStep,
    ToolStep,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowOrchestrator,
    WorkflowResult,
)
from design_research_agents.contracts.tools import ToolRuntime

from .workflow_runtime import WorkflowRuntime

DEFAULT_INVENTORY_CSV_PATH = "artifacts/examples/workflow_tool_inventory.csv"
DEFAULT_INVENTORY_CSV_CONTENT = (
    "tool,source\n"
    "calculator,core\n"
    "search.ripgrep,core\n"
    "repo_quickscan,lazy\n"
    "local_core::calculator,mcp\n"
)
DEFAULT_SEARCH_ROOT = "src/design_research_agents/tools"
DEFAULT_SEARCH_QUERY = "UnifiedToolRuntime"
DEFAULT_SEARCH_MAX_MATCHES = 6


def _dependency_result_payload(
    context: Mapping[str, object],
    dependency_step_id: str,
) -> Mapping[str, object]:
    """Return one dependency ``output.result`` payload from workflow context."""
    raw_dependency_results = context.get("dependency_results")
    if not isinstance(raw_dependency_results, Mapping):
        raise KeyError("Workflow context is missing 'dependency_results'.")
    raw_dependency_entry = raw_dependency_results.get(dependency_step_id)
    if not isinstance(raw_dependency_entry, Mapping):
        raise KeyError(f"Dependency '{dependency_step_id}' is unavailable in workflow context.")
    raw_output = raw_dependency_entry.get("output")
    if not isinstance(raw_output, Mapping):
        raise KeyError(f"Dependency '{dependency_step_id}' did not provide an 'output' payload.")
    raw_result = raw_output.get("result")
    if not isinstance(raw_result, Mapping):
        raise KeyError(f"Dependency '{dependency_step_id}' output did not include a result object.")
    return raw_result


def build_pure_tool_workflow_steps(
    *,
    inventory_csv_path: str = DEFAULT_INVENTORY_CSV_PATH,
    inventory_csv_content: str = DEFAULT_INVENTORY_CSV_CONTENT,
    search_root: str = DEFAULT_SEARCH_ROOT,
    search_query: str = DEFAULT_SEARCH_QUERY,
    search_max_matches: int = DEFAULT_SEARCH_MAX_MATCHES,
) -> Sequence[ToolStep | LogicStep]:
    """Build workflow steps for a pure tool+logic orchestration."""
    return [
        ToolStep(
            step_id="seed_csv",
            tool_name="fs.write_text",
            input_data={
                "path": inventory_csv_path,
                "content": inventory_csv_content,
                "overwrite": True,
            },
        ),
        ToolStep(
            step_id="describe_csv",
            tool_name="data.describe",
            dependencies=("seed_csv",),
            input_builder=lambda context: {
                "path": _dependency_result_payload(context, "seed_csv")["path"],
                "kind": "csv",
            },
        ),
        ToolStep(
            step_id="scan_sources",
            tool_name="search.ripgrep",
            dependencies=("describe_csv",),
            input_builder=lambda context: {
                "query": search_query,
                "root": search_root,
                "max_matches": search_max_matches,
            },
        ),
        LogicStep(
            step_id="finalize",
            dependencies=("describe_csv", "scan_sources"),
            handler=lambda context: {
                "csv_rows": _dependency_result_payload(context, "describe_csv")["rows"],
                "csv_columns": _dependency_result_payload(context, "describe_csv")["columns"],
                "runtime_reference_hits": _dependency_result_payload(context, "scan_sources")[
                    "count"
                ],
            },
        ),
    ]


class PureToolWorkflowOrchestrator:
    """Configured pure-tool workflow orchestrator with fixed step topology."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime,
        inventory_csv_path: str = DEFAULT_INVENTORY_CSV_PATH,
        inventory_csv_content: str = DEFAULT_INVENTORY_CSV_CONTENT,
        search_root: str = DEFAULT_SEARCH_ROOT,
        search_query: str = DEFAULT_SEARCH_QUERY,
        search_max_matches: int = DEFAULT_SEARCH_MAX_MATCHES,
    ) -> None:
        """Store runtime dependencies and freeze the generated workflow steps."""
        self._runtime = WorkflowRuntime(tool_runtime=tool_runtime)
        self._steps = build_pure_tool_workflow_steps(
            inventory_csv_path=inventory_csv_path,
            inventory_csv_content=inventory_csv_content,
            search_root=search_root,
            search_query=search_query,
            search_max_matches=search_max_matches,
        )

    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode = "sequential",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute the configured pure-tool workflow."""
        return self._runtime.run(
            self._steps,
            context=context,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            request_id=request_id,
            dependencies=dependencies,
        )


def pure_tool_workflow(
    *,
    tool_runtime: ToolRuntime,
    inventory_csv_path: str = DEFAULT_INVENTORY_CSV_PATH,
    inventory_csv_content: str = DEFAULT_INVENTORY_CSV_CONTENT,
    search_root: str = DEFAULT_SEARCH_ROOT,
    search_query: str = DEFAULT_SEARCH_QUERY,
    search_max_matches: int = DEFAULT_SEARCH_MAX_MATCHES,
) -> WorkflowOrchestrator:
    """Return a configured pure-tool workflow orchestration chunk."""
    return PureToolWorkflowOrchestrator(
        tool_runtime=tool_runtime,
        inventory_csv_path=inventory_csv_path,
        inventory_csv_content=inventory_csv_content,
        search_root=search_root,
        search_query=search_query,
        search_max_matches=search_max_matches,
    )


__all__ = [
    "DEFAULT_INVENTORY_CSV_CONTENT",
    "DEFAULT_INVENTORY_CSV_PATH",
    "DEFAULT_SEARCH_MAX_MATCHES",
    "DEFAULT_SEARCH_QUERY",
    "DEFAULT_SEARCH_ROOT",
    "PureToolWorkflowOrchestrator",
    "build_pure_tool_workflow_steps",
    "pure_tool_workflow",
]
