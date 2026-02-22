"""Run traced ``MultiStepAgent(mode="direct")`` for design-brief drafting.

Expected observations:
- ``steps_executed`` shows CONTINUE/STOP controller progression.
- ``final_output`` contains finalized brief text.
- ``trace.trace_path`` points to emitted run trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Tracer


def main() -> None:
    """Execute one multi-step direct run and print summary."""
    request_id = "example-multi-step-direct-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    try:
        agent = MultiStepAgent(
            mode="direct",
            llm_client=llm_client,
            max_steps=3,
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "Draft then finalize a short design memo title for reducing maintenance time in a modular lab rig."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    payload = {
        "example": "agents/basic/multi_step_direct_llm_agent.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "steps_executed": result.output_value("steps_executed"),
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
