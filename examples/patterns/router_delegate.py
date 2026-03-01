"""# Patterns / Router Delegate.

## Introduction
RouteLLM motivates specialized route selection, AutoGen demonstrates multi-agent delegation patterns, and
Human-AI collaboration by design frames why explicit routing supports accountable coordination. This example
shows intent-based routing across direct and multi-step agents using a shared runtime surface.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``RouterDelegatePattern.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["RouterDelegatePattern.run(...)"]
    C --> D["router delegates to specialized agent surfaces"]
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
- `RouteLLM <https://arxiv.org/abs/2406.18665>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DirectLLMCall,
    LlamaCppServerLLMClient,
    MultiStepAgent,
    Toolbox,
    Tracer,
)
from design_research_agents.patterns import RouterDelegatePattern

_EXAMPLE_LLAMA_CLIENT_KWARGS = {
    "model": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "hf_model_repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "api_model": "qwen3-4b-instruct-2507-q4km",
    "context_window": 8192,
    "startup_timeout_seconds": 240.0,
    "request_timeout_seconds": 240.0,
}


def main() -> None:
    """Route one design prompt to the best delegate and print summary."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-workflow-router-delegate-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    with Toolbox() as tool_runtime, LlamaCppServerLLMClient(**_EXAMPLE_LLAMA_CLIENT_KWARGS) as llm_client:
        direct_llm_agent = DirectLLMCall(llm_client=llm_client, tracer=tracer)
        json_tool_agent = MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            allowed_tools=("text.word_count",),
            tracer=tracer,
        )

        workflow = RouterDelegatePattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            alternatives={
                "direct_llm_agent": direct_llm_agent,
                "json_tool_agent": json_tool_agent,
            },
            alternative_descriptions={
                "direct_llm_agent": "Use for concise textual design summaries with no runtime tools.",
                "json_tool_agent": ("Use for design requests needing runtime text analysis or tool calls."),
            },
            tracer=tracer,
        )

        result = workflow.run(
            prompt=(
                "Count the words in the phrase 'modular field service workflow' using the "
                "appropriate delegate and return only the word_count."
            ),
            request_id=request_id,
        )

    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
