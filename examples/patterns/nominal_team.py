"""# Patterns / Nominal Team.

## Introduction
Nominal teams explore one task independently, then hand all candidate outputs to a dedicated
evaluator for best-of-N selection. This example fans out a design prompt to three focused
contributors and selects the strongest result with a structured evaluator response.

.. note::

   This example's checked-in local ``LlamaCppServerLLMClient`` config uses a
   ``Qwen3-4B`` GGUF model. On lower-RAM machines, swap in a smaller local
   model or start with :doc:`../clients/ollama_local_client`.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build three focused ``DirectLLMCall`` delegates and one evaluator over a shared ``LlamaCppServerLLMClient``.
3. Execute ``NominalTeamPattern.run(...)`` with member-specific prompt templates for diverse independent drafts.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["NominalTeamPattern.run(...)"]
    B --> C["repairability / reliability / manufacturability members generate independently"]
    C --> D["evaluator compares candidates and selects best member"]
    D --> E["ExecutionResult/payload"]
    E --> F["Printed JSON output"]
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<selected-candidate-payload>",
     "terminated_reason": "<string-or-null>",
     "error": null,
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }

## References
- `Self-Consistency Improves Chain of Thought Reasoning in Language Models <https://arxiv.org/abs/2203.11171>`_
- `Tree of Thoughts: Deliberate Problem Solving with Large Language Models <https://arxiv.org/abs/2305.10601>`_
- `Nominal group technique <https://en.wikipedia.org/wiki/Nominal_group_technique>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient, Tracer
from design_research_agents.patterns import NominalTeamPattern

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
    """Run one nominal-team workflow and print JSON summary."""
    request_id = "example-pattern-nominal-team-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    with LlamaCppServerLLMClient(**_EXAMPLE_LLAMA_CLIENT_KWARGS) as llm_client:
        repairability = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=(
                "You are a repairability-focused designer. Return concise JSON with concept, strengths, and risks."
            ),
            tracer=tracer,
        )
        reliability = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=(
                "You are a reliability-focused designer. Return concise JSON with concept, strengths, and risks."
            ),
            tracer=tracer,
        )
        manufacturability = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=(
                "You are a manufacturability-focused designer. Return concise JSON with concept, strengths, and risks."
            ),
            tracer=tracer,
        )
        evaluator = DirectLLMCall(
            llm_client=llm_client,
            system_prompt=(
                "Compare the candidate concepts and return JSON with best_member_id, "
                "scores keyed by member id, and a short rationale."
            ),
            tracer=tracer,
        )

        pattern = NominalTeamPattern(
            team_members=(
                NominalTeamPattern.MemberSpec(
                    member_id="repairability",
                    delegate=repairability,
                    prompt_template=(
                        "Task: {task}\nPerspective: maximize field-service speed and tool simplicity.\n"
                        "Return concise JSON candidate output."
                    ),
                ),
                NominalTeamPattern.MemberSpec(
                    member_id="reliability",
                    delegate=reliability,
                    prompt_template=(
                        "Task: {task}\nPerspective: maximize sealing reliability and failure tolerance.\n"
                        "Return concise JSON candidate output."
                    ),
                ),
                NominalTeamPattern.MemberSpec(
                    member_id="manufacturability",
                    delegate=manufacturability,
                    prompt_template=(
                        "Task: {task}\nPerspective: maximize fabrication simplicity and repeatability.\n"
                        "Return concise JSON candidate output."
                    ),
                ),
            ),
            evaluator_delegate=evaluator,
            tracer=tracer,
        )

        result = pattern.run(
            "Propose a field-serviceable enclosure concept for a remote environmental sensor.",
            request_id=request_id,
        )
    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
