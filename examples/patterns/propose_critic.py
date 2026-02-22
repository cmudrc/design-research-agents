r"""# Patterns / Propose Critic.

## Introduction
Reflexion and Self-Refine motivate iterative self-critique loops, and Human-AI collaboration by design
explains why critique transparency is critical for trustworthy engineering decisions. This example
demonstrates a propose-critic refinement cycle with bounded iterations and structured run output.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``ReflexionPattern.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["ReflexionPattern.run(...)"]
    C --> D["proposal and critique turns iterate until stop criteria"]
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
     "approved": true,
     "critique_iterations": [
       {
         "approved": false,
         "feedback": "Add more detail.",
         "iteration": 1,
         "proposal": "Draft v1: simple proposal.",
         "revision_goals": [
           "expand rationale"
         ]
       },
       {
         "approved": true,
         "feedback": "Looks good.",
         "iteration": 2,
         "proposal": "Draft v2: proposal with more detail.",
         "revision_goals": []
       }
     ],
     "error": null,
     "example": "patterns/propose_critic.py",
     "final_output": null,
     "proposal": "Draft v2: proposal with more detail.",
     "success": true,
     "terminated_reason": "approved",
     "trace": {
       "request_id": "example-workflow-propose-critic-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-workflow-propose-critic-design-001.jsonl"
     }
   }


## References
- `Reflexion <https://arxiv.org/abs/2303.11366>`_
- `Self-Refine <https://arxiv.org/abs/2303.17651>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, Toolbox, Tracer
from design_research_agents.patterns import ReflexionPattern


def main() -> None:
    """Run propose/critique refinement orchestration with tracing."""
    request_id = "example-workflow-propose-critic-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        workflow = ReflexionPattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            tracer=tracer,
        )
        result = workflow.run(
            prompt=(
                "Write and iteratively improve a short engineering design rationale for using "
                "modular connectors in field-serviceable devices."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    payload = {
        "example": "patterns/propose_critic.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "final_output": result.final_output,
        "approved": result.output_value("approved"),
        "critique_iterations": result.output_value("critique_iterations"),
        "proposal": result.output_value("proposal"),
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
