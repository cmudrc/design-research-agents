"""Run traced ``MultiStepAgent(mode="json")`` with memory for design context.

Expected observations:
- ``memory_items`` is non-zero when memory retrieval/write-back is active.
- ``tool_results_count`` indicates tool invocation inside the loop.
- ``trace.trace_path`` points to emitted run trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Toolbox, Tracer
from design_research_agents.memory import SQLiteMemoryStore


def main() -> None:
    """Run one multi-step JSON tool call with memory retrieval and write-back."""
    request_id = "example-multi-step-json-memory-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    db_path = Path("artifacts/examples/multi_step_json_with_memory.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    tool_runtime = Toolbox()
    tool_runtime.invoke_dict(
        "memory.write",
        {
            "db_path": str(db_path),
            "namespace": "design_examples",
            "records": [
                {
                    "content": (
                        "Prior design note: target quick maintenance by minimizing tool changes and "
                        "favoring reusable fasteners."
                    )
                }
            ],
        },
        request_id=f"{request_id}:seed_memory",
        dependencies={},
    )
    store = SQLiteMemoryStore(db_path=db_path)
    llm_client = LlamaCppServerLLMClient()
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
        memory_store=store,
        memory_namespace="design_examples",
        memory_read_top_k=3,
        memory_write_observations=True,
        tracer=tracer,
    )
    try:
        result = agent.run(
            "Compute one design-check text metric and retain the observation history.",
            request_id=request_id,
        )
    finally:
        llm_client.close()
        tool_runtime.close()
        store.close()
    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "agents/basic/multi_step_json_with_memory.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "steps_executed": output.get("steps_executed"),
        "memory_items": (len(output.get("memory", [])) if isinstance(output.get("memory"), list) else 0),
        "tool_results_count": len(result.tool_results),
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
