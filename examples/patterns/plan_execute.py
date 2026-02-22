r"""# Patterns / Plan Execute.

## Introduction
Plan-and-Solve and ReAct both separate planning from execution to reduce reasoning drift, while AutoGen
shows how these roles can be modularized across components. This example encodes planner-executor separation
with tool-backed execution and deterministic trace artifacts.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``PlannerExecutorPattern.run(...)`` with a fixed
   ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["PlannerExecutorPattern.run(...)"]
    C --> D["Planner and executor phases share tool/runtime state"]
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
     "error": null,
     "example": "patterns/plan_execute.py",
     "final_output": {
       "column_count": 2,
       "csv_path": "artifacts/examples/plan_execute_runtime_inventory.csv",
       "row_count": 3,
       "search_hits": 4
     },
     "plan_step_count": 1,
     "steps_executed": 1,
     "success": true,
     "terminated_reason": "completed",
     "trace": {
       "request_id": "example-workflow-plan-execute-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-workflow-plan-execute-design-001.jsonl"
     }
   }


## References
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from design_research_agents import (
    CallableTool,
    LlamaCppServerLLMClient,
    PlannerExecutorPattern,
    Toolbox,
    Tracer,
)


def _readme_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    """Return basic README metrics for plan/execute demos."""
    del payload
    readme_path = Path("README.md")
    readme_text = readme_path.read_text(encoding="utf-8")
    lines = readme_text.splitlines()
    first_heading = next((line.lstrip("#").strip() for line in lines if line.startswith("#")), "")
    return {
        "path": str(readme_path),
        "line_count": len(lines),
        "first_heading": first_heading,
    }


def main() -> None:
    """Run planner-executor orchestration with tracing."""
    request_id = "example-workflow-plan-execute-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox(
        callable_tools=(
            CallableTool(
                name="repo.readme_metrics",
                description="Return README line-count and first heading.",
                handler=_readme_metrics,
            ),
        ),
    )
    try:
        workflow = PlannerExecutorPattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_iterations=1,
            tracer=tracer,
        )
        result = workflow.run(
            prompt=(
                "Create and execute a concise engineering-design audit plan for repository "
                "tooling surfaces and produce a compact summary."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    plan_payload = result.output_dict("plan")
    plan_steps = plan_payload.get("steps") if isinstance(plan_payload, dict) else None
    payload = {
        "example": "patterns/plan_execute.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "steps_executed": result.output_value("steps_executed"),
        "plan_step_count": len(plan_steps) if isinstance(plan_steps, list) else 0,
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
