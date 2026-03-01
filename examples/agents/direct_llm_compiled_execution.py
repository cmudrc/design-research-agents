"""# Agents / Direct LLM Compiled Execution.

## Introduction
Compiled delegate execution is useful when you want to inspect the bound workflow and tracing metadata
before running it. This example makes the intermediate ``CompiledExecution`` object explicit so the compile
step itself is visible and testable.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``DirectLLMCall.compile(...)`` to obtain
   ``CompiledExecution``.
3. Validate the compiled wrapper shape, then call ``CompiledExecution.run()`` with the bound request metadata.
4. Print a compact JSON payload that includes compile metadata alongside the normalized execution summary.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["DirectLLMCall.compile(...)"]
    C --> D["CompiledExecution"]
    D --> E["CompiledExecution.run()"]
    E --> F["ExecutionResult/payload"]
    F --> G["Printed JSON output"]
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "compiled_delegate_name": "DirectLLMCall",
     "compiled_request_id": "<request-id>",
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
- `Partial evaluation (Wikipedia) <https://en.wikipedia.org/wiki/Partial_evaluation>`_
- `Builder pattern (Wikipedia) <https://en.wikipedia.org/wiki/Builder_pattern>`_
- `Interpreter pattern (Wikipedia) <https://en.wikipedia.org/wiki/Interpreter_pattern>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    CompiledExecution,
    DirectLLMCall,
    LlamaCppServerLLMClient,
    Tracer,
)


def main() -> None:
    """Compile one direct model call, then run the compiled execution."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-direct-llm-compiled-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    # Run the compiled direct LLM example using public runtime APIs. Using this with statement will automatically
    # shut down the managed client when the example is done.
    with LlamaCppServerLLMClient() as llm_client:
        llm = DirectLLMCall(llm_client=llm_client, tracer=tracer)
        prompt = (
            "Write one sentence describing the one primary engineering specification for a "
            "field-repairable wearable sensor enclosure."
        )
        compiled: CompiledExecution = llm.compile(
            prompt=prompt,
            request_id=request_id,
        )
        result = compiled.run()

    # Print the results
    payload = {
        "compiled_delegate_name": compiled.delegate_name,
        "compiled_request_id": compiled.request_id,
        **result.summary(),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
