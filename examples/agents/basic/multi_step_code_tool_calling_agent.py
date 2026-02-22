"""Run traced ``MultiStepAgent(mode="code")`` for design-metric analysis.

Expected observations:
- ``step_outputs_count`` is non-zero for iterative code-action execution.
- ``tool_results_count`` shows runtime tool usage.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Toolbox, Tracer


def main() -> None:
    """Execute one multi-step code-mode run and print compact result."""
    request_id = "example-multi-step-code-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        agent = MultiStepAgent(
            mode="code",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            normalize_generated_code_per_step=True,
            default_tools_per_step=({"tool_name": "text.word_count"},),
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "No imports. Use call_tool only. Compute two design-review metrics using "
                "text.word_count on these phrases: 'design review metrics' and "
                "'runtime tool boundaries'. Return final_output with both counts."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "agents/basic/multi_step_code_tool_calling_agent.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "steps_executed": output.get("steps_executed"),
        "step_outputs_count": len(output.get("step_outputs", []))
        if isinstance(output.get("step_outputs"), list)
        else 0,
        "tool_results_count": len(result.tool_results),
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
