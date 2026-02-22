"""Run traced ``DebatePattern`` on an engineering design tradeoff.

Expected observations:
- ``rounds`` and ``winner`` summarize debate outcomes.
- ``verdict`` captures judge synthesis.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import DebatePattern, LlamaCppServerLLMClient, Toolbox, Tracer


def main() -> None:
    """Run one debate round with final judge verdict."""
    request_id = "example-workflow-debate-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        workflow = DebatePattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_rounds=1,
            tracer=tracer,
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

    payload = {
        "example": "workflow/debate_pattern.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "final_output": result.final_output,
        "rounds": result.output_value("rounds"),
        "winner": result.output_value("winner"),
        "verdict": result.output_value("verdict"),
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
