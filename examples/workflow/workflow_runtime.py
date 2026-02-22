"""Run traced public ``Workflow`` facade for a design-check summary.

Expected observations:
- ``success`` is ``True`` for the deterministic logic-only run.
- ``final_output`` includes a compact design-runtime readiness message.
- ``trace.trace_path`` points to emitted trace JSONL.
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
