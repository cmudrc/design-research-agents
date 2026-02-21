"""Run traced ``MultiStepAgent(mode="json")`` for design-risk scoring.

Expected observations:
- ``tool_results_count`` confirms tool-boundary execution.
- ``final_output`` includes model summary for the design-risk check.
- ``trace.trace_path`` points to trace JSONL artifact.
"""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents import CallableTool, LlamaCppServerLLMClient, MultiStepAgent, Toolbox
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def _risk_score(payload: Mapping[str, object]) -> dict[str, object]:
    """Return deterministic risk score for a design concept."""
    concept = str(payload.get("concept", "quick-release hinge"))
    score = 0.22 if "quick-release" in concept else 0.41
    return {
        "concept": concept,
        "risk_score": score,
        "note": "Lower scores indicate lower implementation risk.",
    }


def main() -> None:
    """Execute one traced multi-step JSON tool-calling run."""
    request_id = "example-multi-step-json-design-001"
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableTool(
                name="design.risk_score",
                description="Return a deterministic design implementation risk score.",
                handler=_risk_score,
            ),
        ),
    )
    try:
        agent = MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=1,
            tracer=make_tracer(),
        )
        result = agent.run(
            prompt=(
                "Call design.risk_score for concept 'quick-release hinge with modular gasket' "
                "and summarize the risk."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "agents/basic/multi_step_json_tool_calling_agent.py",
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "steps_executed": output.get("steps_executed"),
        "tool_results_count": len(result.tool_results),
        "final_output": output.get("final_output"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
