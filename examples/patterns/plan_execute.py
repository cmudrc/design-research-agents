"""Example script.

Motivation
Run traced ``PlannerExecutorPattern`` for an engineering design audit task.

Diagram
```mermaid
flowchart LR
    A["Pattern prompt"] --> B["Pattern orchestration"]
    B --> C["plan execute result"]
    C --> D["Trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `patterns` use-cases and run `plan_execute`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/patterns/plan_execute.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
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
