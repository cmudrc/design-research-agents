"""# Patterns / Plan Execute.

## Introduction
Plan-and-Solve and ReAct both separate planning from execution to reduce reasoning drift, while AutoGen
shows how these roles can be modularized across components. This example encodes planner-executor separation
with tool-backed execution and deterministic trace artifacts.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``PlanExecutePattern.run(...)`` with a fixed
   ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["PlanExecutePattern.run(...)"]
    C --> D["Planner and executor phases share tool/runtime state"]
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
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    LlamaCppServerLLMClient,
    MultiStepAgent,
    Toolbox,
    Tracer,
)
from design_research_agents.patterns import PlanExecutePattern

_EXAMPLE_LLAMA_CLIENT_KWARGS = {
    "model": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "hf_model_repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "api_model": "qwen3-4b-instruct-2507-q4km",
    "context_window": 8192,
    "startup_timeout_seconds": 240.0,
    "request_timeout_seconds": 240.0,
}


def main() -> None:
    """Run planner-executor orchestration with tracing."""
    # Fixed request ids keep trace paths and sample output stable for docs/tests.
    request_id = "example-workflow-plan-execute-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    # Run the planner/executor pattern using public runtime surfaces. Using this with statement will
    # automatically shut down the managed client and tool runtime when the example is done.
    with Toolbox() as tool_runtime, LlamaCppServerLLMClient(**_EXAMPLE_LLAMA_CLIENT_KWARGS) as llm_client:
        executor_delegate = MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            allowed_tools=("text.word_count",),
            tracer=tracer,
        )
        workflow = PlanExecutePattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            executor_delegate=executor_delegate,
            max_iterations=1,
            tracer=tracer,
        )
        result = workflow.run(
            prompt=(
                "Create and execute a one-step plan that uses text.word_count to count the words "
                "in the phrase 'design system research workflow', then return only word_count."
            ),
            request_id=request_id,
        )

    # Print the results
    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
