r"""# Workflow / Workflow Runtime.

## Introduction
Human-AI collaboration by design motivates transparent orchestration boundaries, AutoGen motivates
composable multi-component execution, and HELM motivates repeatable runtime instrumentation for comparisons.
This example is the minimal workflow-runtime build for observing step execution semantics directly.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["Workflow.run(...)"]
    C --> D["WorkflowRuntime schedules step graph (LogicStep)"]
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
     "example": "workflow/workflow_runtime.py",
     "execution_order": [
       "design_runtime_ready"
     ],
     "final_output": {
       "check": "workflow-runtime-ready",
       "message": "Design runtime orchestration validated."
     },
     "success": true,
     "terminated_reason": null,
     "trace": {
       "request_id": "example-workflow-runtime-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-runtime-design-001.jsonl"
     }
   }


## References
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
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
