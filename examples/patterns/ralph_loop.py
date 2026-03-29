"""# Patterns / Ralph Loop.

## Introduction
Ralph loops are role-programmed, not fixed two-role propose/critic cycles: each round executes
an ordered role lineup, then a dedicated evaluator decides whether consensus quality is high enough.
This example demonstrates a four-role configuration with synthesis selection and threshold stopping.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build role-specific delegates with ``DirectLLMCall`` over one managed ``LlamaCppServerLLMClient``.
3. Execute ``RalphLoopPattern.run(...)`` with dynamic roles, evaluator role id, and typed ``LoopConfig``.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["RalphLoopPattern.run(...)"]
    C --> D["role batch executes proposer/critic/synthesizer/evaluator each round"]
    C --> E["evaluator score compared to consensus threshold"]
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
- `CAMEL: Communicative Agents for Mind Exploration <https://arxiv.org/abs/2303.17760>`_
- `MetaGPT <https://arxiv.org/abs/2308.00352>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient, Tracer
from design_research_agents.patterns import RalphLoopPattern

_EXAMPLE_LLAMA_CLIENT_KWARGS = {
    "model": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "hf_model_repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "api_model": "qwen3-4b-instruct-2507-q4km",
    "context_window": 8192,
    "startup_timeout_seconds": 240.0,
    "request_timeout_seconds": 240.0,
}


def main() -> None:
    """Run one Ralph loop workflow and print JSON summary."""
    request_id = "example-pattern-ralph-loop-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    with LlamaCppServerLLMClient(**_EXAMPLE_LLAMA_CLIENT_KWARGS) as llm_client:
        proposer = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=("You are a design proposer. Return concise JSON with proposal options and intended change."),
            tracer=tracer,
        )
        critic = DirectLLMCall(
            llm_client=llm_client,
            system_prompt="You are a design critic. Return concise JSON with risks and revision advice.",
            tracer=tracer,
        )
        synthesizer = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=(
                "You are a synthesis role. Merge proposal + critique into one implementation-ready JSON summary."
            ),
            tracer=tracer,
        )
        evaluator = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=("You are the evaluator. Return JSON with numeric score in [0,1] and brief rationale."),
            tracer=tracer,
        )

        pattern = RalphLoopPattern(
            roles=(
                RalphLoopPattern.RoleSpec(
                    role_id="proposer",
                    delegate=proposer,
                    prompt_template=(
                        "Task: {task}\nIteration: {iteration}\nCurrent selected output:"
                        " {selected_output_json}\nReturn JSON for the next proposal."
                    ),
                ),
                RalphLoopPattern.RoleSpec(
                    role_id="critic",
                    delegate=critic,
                    prompt_template=(
                        "Task: {task}\nIteration: {iteration}\nPrior role outputs:"
                        " {prior_role_outputs_json}\nReturn JSON critique for the proposer."
                    ),
                ),
                RalphLoopPattern.RoleSpec(
                    role_id="synthesizer",
                    delegate=synthesizer,
                    prompt_template=(
                        "Task: {task}\nIteration: {iteration}\nPrior role outputs:"
                        " {prior_role_outputs_json}\nReturn JSON synthesis ready for evaluation."
                    ),
                ),
                RalphLoopPattern.RoleSpec(
                    role_id="evaluator",
                    delegate=evaluator,
                    prompt_template=(
                        "Task: {task}\nIteration: {iteration}\nCandidate synthesis:"
                        " {selected_output_json}\nRole outputs: {prior_role_outputs_json}\n"
                        "Return JSON with score in [0,1]."
                    ),
                ),
            ),
            evaluator_role_id="evaluator",
            loop_config=RalphLoopPattern.LoopConfig(
                max_iterations=3,
                consensus_threshold=0.8,
                selection_strategy="best_score",
            ),
            tracer=tracer,
        )

        result = pattern.run(
            "Refine a field-serviceable edge-device enclosure concept.",
            request_id=request_id,
        )
    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
