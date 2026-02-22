"""Run traced ``MultiStepAgent(mode="json")`` with explicit core-tool allowlist.

Expected observations:
- ``tool_results_count`` confirms tool-boundary execution.
- ``allowed_tools`` documents recommended scoped JSON-agent tool access.
- ``trace.trace_path`` points to trace JSONL artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Toolbox, Tracer

_JSON_ALLOWED_TOOLS: tuple[str, ...] = (
    "fs.read_text",
    "text.word_count",
    "python.sandbox",
    "memory.search",
    "memory.write",
    "memory.stats",
    "eval.decision_matrix",
    "eval.pairwise_rank",
)


def main() -> None:
    """Execute one traced multi-step JSON tool-calling run."""
    request_id = "example-multi-step-json-design-001"
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
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            allowed_tools=_JSON_ALLOWED_TOOLS,
            tracer=tracer,
        )
        result = agent.run(
            prompt="Read README.md and summarize one implementation insight from the text.",
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "agents/basic/multi_step_json_tool_calling_agent.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "steps_executed": output.get("steps_executed"),
        "tool_results_count": len(result.tool_results),
        "allowed_tools": list(_JSON_ALLOWED_TOOLS),
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
