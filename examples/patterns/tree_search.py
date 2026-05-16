"""# Patterns / Tree Search.

## Introduction
Tree of Thoughts motivates explicit branching and ranking instead of single-pass revision.
This example uses dedicated generator/evaluator delegates and a bounded beam search to show
search-policy behavior (expand, score, prune) in a traceable way.

.. note::

   This example's checked-in local ``LlamaCppServerLLMClient`` config uses a
   ``Qwen3-4B`` GGUF model. On lower-RAM machines, swap in a smaller local
   model or start with :doc:`../clients/ollama_local_client`.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build generator and evaluator delegates with ``DirectLLMCall`` and a managed ``LlamaCppServerLLMClient``.
3. Execute ``TreeSearchPattern.run(...)`` with explicit search controls and preserve frontier diagnostics.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["TreeSearchPattern.run(...)"]
    C --> D["generator/evaluator delegates expand and score candidate nodes"]
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
- `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient, Tracer
from design_research_agents.patterns import TreeSearchPattern

# This checked-in local config uses a Qwen3-4B GGUF model to exercise a richer
# multi-step path. On lower-RAM machines, swap in a smaller local model or
# start with the lighter Ollama local client example first.
_EXAMPLE_LLAMA_CLIENT_KWARGS = {
    "model": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "hf_model_repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "api_model": "qwen3-4b-instruct-2507-q4km",
    "context_window": 8192,
    "startup_timeout_seconds": 240.0,
    "request_timeout_seconds": 240.0,
}


def main() -> None:
    """Run one tree-search workflow and print JSON summary."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-pattern-tree-search-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    with LlamaCppServerLLMClient(**_EXAMPLE_LLAMA_CLIENT_KWARGS) as llm_client:
        generator_delegate = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=(
                "You are a search-node generator. Return JSON with key `candidates` mapped to a list of"
                " 1-2 short candidate objects. Keep output concise."
            ),
            tracer=tracer,
        )
        evaluator_delegate = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=(
                "You are a search-node evaluator. Return JSON with numeric key `score` in [0,1]"
                " for the candidate provided by the user."
            ),
            tracer=tracer,
        )
        pattern = TreeSearchPattern(
            generator_delegate=generator_delegate,
            evaluator_delegate=evaluator_delegate,
            max_depth=2,
            branch_factor=2,
            beam_width=1,
            search_strategy="beam",
            tracer=tracer,
        )
        result = pattern.run(
            "Find the most robust concept architecture for a serviceable edge-device enclosure.",
            request_id=request_id,
        )
    # Print the results
    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
