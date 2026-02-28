"""# Agents / Multi Step Code Tool Calling Agent.

## Introduction
ReAct and Toolformer motivate external action for model reasoning, while AutoGen highlights how
multi-agent/tool ecosystems depend on explicit execution boundaries. This example focuses on code-tool
calling so you can study how executable outputs are requested, validated, and traced in a controlled loop.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
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
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Toolbox, Tracer


def main() -> None:
    """Execute one multi-step code-mode run and print compact result."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-multi-step-code-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    with Toolbox() as tool_runtime, LlamaCppServerLLMClient() as llm_client:
        agent = MultiStepAgent(
            mode="code",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            normalize_generated_code_per_step=True,
            default_tools_per_step=({"tool_name": "text.word_count"},),
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "No imports. Use call_tool only. Compute two design-review metrics using "
                "text.word_count on these phrases: 'design review metrics' and "
                "'runtime tool boundaries'. Return final_output with both counts."
            ),
            request_id=request_id,
        )

    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
