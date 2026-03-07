"""# Agents / Direct LLM Call.

## Introduction
The default built-in path is the OpenAI-compatible HTTP client. This keeps the base install lightweight
while still talking to a real endpoint, whether that endpoint is local (for example llama.cpp, vLLM, or
SGLang) or remote behind an OpenAI-shaped gateway.


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
- `OpenAI API Reference <https://platform.openai.com/docs/api-reference/chat>`_
- `llama.cpp server documentation <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag


def main() -> None:
    """Execute one direct model call with explicit tracing."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-direct-llm-design-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )

    # Point the built-in HTTP client at any reachable OpenAI-compatible endpoint.
    with drag.OpenAICompatibleHTTPLLMClient(
        base_url="http://127.0.0.1:8001/v1",
        default_model="qwen2-1.5b-q4",
    ) as llm_client:
        llm = drag.DirectLLMCall(llm_client=llm_client, tracer=tracer)
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
