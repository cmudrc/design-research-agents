"""# Agents / Direct LLM Call.

## Introduction
Fast offline onboarding is easiest when the runtime path is fully inspectable, deterministic, and free of
external service setup. This example uses the built-in HTML stand-in client so a first run still exercises
the same tracing and execution contracts without requiring network access, model downloads, or API keys.


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
- `WHATWG HTML Living Standard <https://html.spec.whatwg.org/>`_
- `Python html module <https://docs.python.org/3/library/html.html>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DirectLLMCall,
    HTMLLLMClient,
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

    # Run the direct LLM call using the zero-dependency HTML stand-in client.
    with HTMLLLMClient() as llm_client:
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
