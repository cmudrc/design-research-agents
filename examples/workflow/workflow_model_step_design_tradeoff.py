"""Run traced ``ModelStep`` workflow for deterministic design tradeoff text.

Expected observations:
- ``ModelStep`` executes an LLM request via a deterministic local client.
- ``response_parser`` produces structured tradeoff output.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import LogicStep, ModelStep, Workflow
from design_research_agents.contracts import LLMMessage, LLMRequest
from design_research_agents.shared.deterministic_design_helpers import (
    DeterministicSequenceLLMClient,
)
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Run model-step workflow and print compact design tradeoff summary."""
    request_id = "example-workflow-model-step-design-001"
    llm_client = DeterministicSequenceLLMClient(
        responses=[
            "Use a modular latch for faster maintenance; accept small cost "
            "increase for serviceability."
        ]
    )

    workflow = Workflow(
        tool_runtime=None,
        tracer=make_tracer(),
        input_mode="schema",
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
                    "tradeoff": context["dependency_results"]["design_tradeoff_model"]["output"][
                        "parsed"
                    ]["tradeoff_summary"]
                },
            ),
        ],
    )

    result = workflow.run(
        {"design_goal": "reduce repair time for edge-device battery modules"},
        request_id=request_id,
    )
    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "workflow/workflow_model_step_design_tradeoff.py",
        "success": result.success,
        "final_output": output.get("final_output"),
        "execution_order": list(result.execution_order),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
