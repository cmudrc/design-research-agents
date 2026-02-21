"""Run traced ``DebatePattern`` on an engineering design tradeoff.

Expected observations:
- ``rounds`` and ``winner`` summarize debate outcomes.
- ``verdict`` captures judge synthesis.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import DebatePattern, LlamaCppServerLLMClient, Toolbox
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Run one debate round with final judge verdict."""
    request_id = "example-workflow-debate-design-001"
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        workflow = DebatePattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_rounds=1,
            tracer=make_tracer(),
        )
        result = workflow.run(
            prompt=(
                "Should an engineering design team prioritize local models over hosted APIs when "
                "reviewing sensitive prototype telemetry?"
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "workflow/debate_pattern.py",
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "rounds": output.get("rounds"),
        "winner": output.get("winner"),
        "verdict": output.get("verdict"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
