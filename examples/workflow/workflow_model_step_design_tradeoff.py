"""# Workflow / Workflow Model Step Design Tradeoff.

## Introduction
FrugalGPT frames cost-aware model choice, HELM frames robust comparative evaluation, and Toward Engineering
AGI frames engineering-task relevance of those choices. This example demonstrates model-step tradeoff
handling inside a workflow graph with deterministic trace capture.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.


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
- `FrugalGPT <https://arxiv.org/abs/2305.05176>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
- `Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs <https://arxiv.org/abs/2509.16204>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag

WORKFLOW_DIAGRAM_DIRECTION = "LR"


class _DocLLMClient:
    """Minimal LLM client stub used only for docs-diagram workflow construction."""

    def default_model(self) -> str:
        return "doc-model"


def build_example_workflow(
    *,
    tracer: drag.Tracer | None = None,
    llm_client: object | None = None,
) -> drag.Workflow:
    """Build the model-step workflow used for docs diagrams and runtime execution."""
    resolved_llm_client = llm_client or _DocLLMClient()
    return drag.Workflow(
        tool_runtime=None,
        tracer=tracer,
        input_schema={"type": "object"},
        steps=[
            drag.ModelStep(
                step_id="design_tradeoff_model",
                llm_client=resolved_llm_client,
                request_builder=lambda context: drag.LLMRequest(
                    messages=[
                        drag.LLMMessage(
                            role="user",
                            content=(
                                "Summarize one engineering tradeoff for this goal: "
                                f"{context['inputs'].get('design_goal', '')}"
                            ),
                        )
                    ],
                    model=resolved_llm_client.default_model(),
                ),
                response_parser=lambda response, _context: {
                    "tradeoff_summary": response.text,
                    "model": response.model,
                },
            ),
            drag.LogicStep(
                step_id="finalize",
                dependencies=("design_tradeoff_model",),
                handler=lambda context: {
                    "tradeoff": context["dependency_results"]["design_tradeoff_model"]["output"]["parsed"][
                        "tradeoff_summary"
                    ]
                },
            ),
        ],
    )


def main() -> None:
    """Run model-step workflow and print compact design tradeoff summary."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-workflow-model-step-design-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    # Run the model-step workflow using public runtime surfaces. Using this with statement will automatically
    # shut down the managed client when the example is done.
    with drag.LlamaCppServerLLMClient() as llm_client:
        workflow = build_example_workflow(tracer=tracer, llm_client=llm_client)

        result = workflow.run(
            {"design_goal": "reduce repair time for edge-device battery modules"},
            request_id=request_id,
        )
    # Print the results
    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
