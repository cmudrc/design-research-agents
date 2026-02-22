"""Run one traced ``DirectLLMCall`` for an engineering-design prompt.

Expected observations:
- ``success`` indicates whether the run completed.
- ``final_output`` contains the model's concise design objective text.
- ``trace.trace_path`` points to a JSONL trace artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient, Tracer, __version__


def main() -> None:
    """Execute one direct model call with explicit tracing."""
    request_id = "example-direct-llm-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    try:
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
        "final_output": result.final_output,
        "model": output.get("model"),
        "terminated_reason": result.terminated_reason,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
