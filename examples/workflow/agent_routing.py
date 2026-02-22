"""Run traced ``RouterPattern`` across design-focused delegate agents.

Expected observations:
- ``selected_alternative`` indicates routed delegate.
- ``final_output`` includes delegated answer payload.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DirectLLMCall,
    LlamaCppServerLLMClient,
    MultiStepAgent,
    RouterPattern,
    Toolbox,
    Tracer,
)


def main() -> None:
    """Route one design prompt to the best delegate and print summary."""
    request_id = "example-workflow-agent-routing-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
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
            "json_tool_agent": ("Use for design requests needing runtime text analysis or tool calls."),
        },
        tracer=tracer,
    )

    try:
        result = workflow.run(
            prompt=("Count the words in this design phrase and return the tool result: modular field service workflow"),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "workflow/agent_routing.py",
        "success": result.success,
        "selected_alternative": output.get("selected_alternative"),
        "final_output": result.final_output,
        "terminated_reason": result.terminated_reason,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
