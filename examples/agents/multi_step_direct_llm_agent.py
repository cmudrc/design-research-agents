"""# Agents / Multi Step Direct LLM Agent.

## Introduction
ReAct and Plan-and-Solve both motivate explicit multi-step reasoning loops instead of single-shot prompting,
and Toward Engineering AGI highlights why that structure matters for measurable engineering outcomes. This
example demonstrates a direct multi-step agent loop with traced iterations so design reasoning can be
inspected rather than inferred.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["MultiStepAgent.run(...)"]
    C --> D["WorkflowRuntime loop enforces continuation and max-step policy"]
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
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs <https://arxiv.org/abs/2509.16204>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Tracer


def main() -> None:
    """Execute one multi-step direct run and print summary."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-multi-step-direct-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    with LlamaCppServerLLMClient() as llm_client:
        agent = MultiStepAgent(
            mode="direct",
            llm_client=llm_client,
            max_steps=3,
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "Draft then finalize a short design memo title for reducing maintenance time in a modular lab rig."
            ),
            request_id=request_id,
        )

    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
