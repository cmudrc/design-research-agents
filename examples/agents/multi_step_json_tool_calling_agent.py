"""# Agents / Multi Step JSON Tool Calling Agent.

## Introduction
Toolformer motivates tool-use planning, JSON Schema defines stable machine-readable contracts, and OpenAI
function-calling guidance captures operational patterns for structured tool dispatch. This example shows a
JSON-mode agent that repeatedly selects tools through explicit schema-constrained payloads.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["MultiStepAgent.run(...)"]
    C --> D["WorkflowRuntime loop enforces explicit final-answer and max-step policy"]
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
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag

_EXAMPLE_LLAMA_CLIENT_KWARGS = {
    "model": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "hf_model_repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "api_model": "qwen3-4b-instruct-2507-q4km",
    "context_window": 8192,
    "startup_timeout_seconds": 240.0,
    "request_timeout_seconds": 240.0,
}


def main() -> None:
    """Execute one traced multi-step JSON tool-calling run."""
    # Stable ids make trace correlation and docs output easier to audit.
    request_id = "example-multi-step-json-design-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    # Run the JSON tool-calling example using public runtime surfaces. Using this with statement will automatically
    # shut down the managed client and tool runtime when the example is done.
    with drag.Toolbox() as tool_runtime, drag.LlamaCppServerLLMClient(**_EXAMPLE_LLAMA_CLIENT_KWARGS) as llm_client:
        json_tool_agent = drag.MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            # Constrain selection so the example exercises an explicit tool surface.
            allowed_tools=("text.word_count",),
            tracer=tracer,
        )
        result = json_tool_agent.run(
            prompt=(
                "Use text.word_count once to count the words in the phrase "
                "'design research agents', then finish by returning only the word_count."
            ),
            request_id=request_id,
        )

    # Print the results
    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
