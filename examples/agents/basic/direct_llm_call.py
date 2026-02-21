"""Run one traced ``DirectLLMCall`` for an engineering-design prompt.

Expected observations:
- ``success`` indicates whether the run completed.
- ``final_output`` contains the model's concise design objective text.
- ``trace.trace_path`` points to a JSONL trace artifact.
"""

from __future__ import annotations

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient, __version__
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Execute one direct model call with explicit tracing."""
    request_id = "example-direct-llm-design-001"
    llm_client = LlamaCppServerLLMClient()
    try:
        tracer = make_tracer()
        agent = DirectLLMCall(llm_client=llm_client, tracer=tracer)
        result = agent.run(
            prompt=(
                "Write one sentence describing the primary engineering objective for a "
                "field-repairable wearable sensor enclosure."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "agents/basic/direct_llm_call.py",
        "package_version": __version__,
        "success": result.success,
        "final_output": output.get("final_output"),
        "model": output.get("model"),
        "terminated_reason": output.get("terminated_reason"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
