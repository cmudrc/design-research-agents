r"""# Patterns / Propose Critic.

## Introduction
Reflexion and Self-Refine motivate iterative self-critique loops, and Human-AI collaboration by design
explains why critique transparency is critical for trustworthy engineering decisions. This example
demonstrates a propose-critic refinement cycle with bounded iterations and structured run output.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``ProposeCriticPattern.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["ProposeCriticPattern.run(...)"]
    C --> D["proposal and critique turns iterate until stop criteria"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<example-specific payload>",
     "terminated_reason": "<string-or-null>",
     "error": null,
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
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
from design_research_agents.patterns import ProposeCriticPattern


def main() -> None:
    """Run propose/critique refinement orchestration with tracing."""
    # Keep request ids deterministic so critique traces are easy to compare run-to-run.
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
        workflow = ProposeCriticPattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            # Tracer is threaded through the pattern so proposer/critic turns share one timeline.
            tracer=tracer,
        )
        result = workflow.run(
            prompt=(
                "Write and iteratively improve a short engineering design rationale for using "
                "modular connectors in field-serviceable devices."
            ),
            request_id=request_id,
        )
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        llm_client.close()

    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
