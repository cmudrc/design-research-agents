"""Example script.

Motivation
Run traced public ``Workflow`` facade for a design-check summary.

Diagram
```mermaid
flowchart LR
    A["Workflow input"] --> B["Workflow steps"]
    B --> C["workflow runtime final output"]
    C --> D["Trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `workflow` use-cases and run `workflow_runtime`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/workflow/workflow_runtime.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LogicStep, Tracer, Workflow


def main() -> None:
    """Run a minimal logic workflow and print literal payload."""
    request_id = "example-workflow-runtime-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    workflow = Workflow(
        tool_runtime=None,
        input_schema={"type": "object"},
        tracer=tracer,
        steps=[
            LogicStep(
                step_id="design_runtime_ready",
                handler=lambda _context: {
                    "message": "Design runtime orchestration validated.",
                    "check": "workflow-runtime-ready",
                },
            )
        ],
    )
    result = workflow.run({}, execution_mode="sequential", request_id=request_id)
    payload = {
        "example": "workflow/workflow_runtime.py",
        "success": result.success,
        "execution_order": list(result.execution_order),
        "final_output": result.final_output,
        "terminated_reason": result.terminated_reason,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
