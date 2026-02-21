"""Run traced ``MultiStepAgent(mode="direct")`` for design-brief drafting.

Expected observations:
- ``steps_executed`` shows CONTINUE/STOP controller progression.
- ``final_output`` contains finalized brief text.
- ``trace.trace_path`` points to emitted run trace JSONL.
"""

from __future__ import annotations

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Execute one multi-step direct run and print summary."""
    request_id = "example-multi-step-direct-design-001"
    llm_client = LlamaCppServerLLMClient()
    try:
        agent = MultiStepAgent(
            mode="direct",
            llm_client=llm_client,
            max_steps=3,
            tracer=make_tracer(),
        )
        result = agent.run(
            prompt=(
                "Draft then finalize a short design memo title for reducing maintenance time in "
                "a modular lab rig."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "agents/basic/multi_step_direct_llm_agent.py",
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "steps_executed": output.get("steps_executed"),
        "final_output": output.get("final_output"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
