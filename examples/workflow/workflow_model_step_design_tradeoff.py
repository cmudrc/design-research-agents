"""Run traced ``ModelStep`` workflow for deterministic design tradeoff text.

Expected observations:
- ``ModelStep`` executes an LLM request via a deterministic local client.
- ``response_parser`` produces structured tradeoff output.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, LogicStep, ModelStep, Tracer, Workflow
from design_research_agents.llm import LLMMessage, LLMRequest


def main() -> None:
    """Run model-step workflow and print compact design tradeoff summary."""
    request_id = "example-workflow-model-step-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()

    try:
        workflow = Workflow(
            tool_runtime=None,
            tracer=tracer,
            input_schema={"type": "object"},
            steps=[
                ModelStep(
                    step_id="design_tradeoff_model",
                    llm_client=llm_client,
                    request_builder=lambda context: LLMRequest(
                        messages=[
                            LLMMessage(
                                role="user",
                                content=(
                                    "Summarize one engineering tradeoff for this goal: "
                                    f"{context['inputs'].get('design_goal', '')}"
                                ),
                            )
                        ],
                        model=llm_client.default_model(),
                    ),
                    response_parser=lambda response, _context: {
                        "tradeoff_summary": response.text,
                        "model": response.model,
                    },
                ),
                LogicStep(
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

        result = workflow.run(
            {"design_goal": "reduce repair time for edge-device battery modules"},
            request_id=request_id,
        )
    finally:
        llm_client.close()
    payload = {
        "example": "workflow/workflow_model_step_design_tradeoff.py",
        "success": result.success,
        "final_output": result.final_output,
        "terminated_reason": result.terminated_reason,
        "execution_order": list(result.execution_order),
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
