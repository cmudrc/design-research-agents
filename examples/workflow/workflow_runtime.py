"""Run traced public ``Workflow`` facade for a design-check summary.

Expected observations:
- ``success`` is ``True`` for the deterministic logic-only run.
- ``final_output`` includes a compact design-runtime readiness message.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import LogicStep, Workflow
from design_research_agents.shared.example_support import make_tracer, trace_info


def main() -> None:
    """Run a minimal logic workflow and print literal payload."""
    request_id = "example-workflow-runtime-design-001"
    workflow = Workflow(
        tool_runtime=None,
        input_mode="schema",
        tracer=make_tracer(),
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
        "success": result.success,
        "execution_order": list(result.execution_order),
        "final_output": (
            result.output.get("final_output") if isinstance(result.output, dict) else None
        ),
        "trace": trace_info(request_id),
    }
    print(payload)


if __name__ == "__main__":
    main()
