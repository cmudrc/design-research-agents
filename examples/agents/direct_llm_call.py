"""# Agents / Direct LLM Call.

## Introduction
Engineering-design studies show that transparent prompt-to-response traces are essential for credible
evaluation and human oversight; the benchmark framing in Toward Engineering AGI and the collaboration
framing in Human-AI collaboration by design both depend on this visibility, while llama.cpp server docs
ground practical local deployment. This example is the smallest reproducible path for observing one direct
call end to end with runtime traces.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``DirectLLMCall.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["DirectLLMCall.run(...)"]
    C --> D["WorkflowRuntime executes one direct call"]
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
- `Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs <https://arxiv.org/abs/2509.16204>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
- `llama.cpp llama-server docs <https://github.com/ggml-org/llama.cpp#llama-server>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DirectLLMCall,
    LlamaCppServerLLMClient,
    Tracer,
)


def main() -> None:
    """Execute one direct model call with explicit tracing."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-direct-llm-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )

    # Run the direct LLM call using a convenience wrapper. Using this with statement will automatically shut down the
    # client when we're done.
    with LlamaCppServerLLMClient() as llm_client:
        llm = DirectLLMCall(llm_client=llm_client, tracer=tracer)
        prompt = (
            "Write one sentence describing the one primary engineering specification for a "
            "field-repairable wearable sensor enclosure."
        )
        result = llm.run(
            prompt=prompt,
            request_id=request_id,
        )

    # Print the results
    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
