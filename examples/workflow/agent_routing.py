"""Run traced ``RouterPattern`` across design-focused delegate agents.

Expected observations:
- ``selected_alternative`` indicates routed delegate.
- ``final_output`` includes delegated answer payload.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import (
    DirectLLMCall,
    LlamaCppServerLLMClient,
    MultiStepAgent,
    RouterPattern,
    Toolbox,
)
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Route one design prompt to the best delegate and print summary."""
    request_id = "example-workflow-agent-routing-design-001"
    tracer = make_tracer()
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()

    direct_llm_agent = DirectLLMCall(llm_client=llm_client, tracer=tracer)
    json_tool_agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=1,
        tracer=tracer,
    )

    workflow = RouterPattern(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        alternatives={
            "direct_llm_agent": direct_llm_agent,
            "json_tool_agent": json_tool_agent,
        },
        alternative_descriptions={
            "direct_llm_agent": "Use for concise textual design summaries with no runtime tools.",
            "json_tool_agent": (
                "Use for design requests needing runtime calculations or tool calls."
            ),
        },
        tracer=tracer,
    )

    try:
        result = workflow.run(
            prompt=(
                "Calculate this design score expression and return the numeric result: 12 * (4 + 1)"
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "workflow/agent_routing.py",
        "success": result.success,
        "selected_alternative": output.get("selected_alternative"),
        "final_output": output.get("final_output"),
        "terminated_reason": output.get("terminated_reason"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
