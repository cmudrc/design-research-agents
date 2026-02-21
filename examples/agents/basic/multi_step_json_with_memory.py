"""Run traced ``MultiStepAgent(mode="json")`` with memory for design context.

Expected observations:
- ``memory_items`` is non-zero when memory retrieval/write-back is active.
- ``tool_results_count`` indicates tool invocation inside the loop.
- ``trace.trace_path`` points to emitted run trace JSONL.
"""

from __future__ import annotations

from pathlib import Path

from design_research_agents import MultiStepAgent, Toolbox
from design_research_agents.contracts import MemoryWriteRecord
from design_research_agents.memory.stores.sqlite_store import SQLiteMemoryStore
from design_research_agents.shared.deterministic_design_helpers import (
    DeterministicSequenceLLMClient,
)
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Run one multi-step JSON tool call with memory retrieval and write-back."""
    request_id = "example-multi-step-json-memory-design-001"
    db_path = Path("artifacts/examples/multi_step_json_with_memory.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    store = SQLiteMemoryStore(db_path=db_path)
    store.write(
        [
            MemoryWriteRecord(
                content=(
                    "Prior design note: target quick maintenance by minimizing tool changes and "
                    "favoring reusable fasteners."
                )
            )
        ],
        namespace="design_examples",
    )

    llm_client = DeterministicSequenceLLMClient(
        responses=[
            '{"continue": true, "thought": "start"}',
            '{"tool_name": "calculator", "tool_input": {"expression": "12 * (4 + 1)"}}',
            '{"continue": false, "thought": "done"}',
        ]
    )
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
        memory_store=store,
        memory_namespace="design_examples",
        memory_read_top_k=3,
        memory_write_observations=True,
        tracer=make_tracer(),
    )
    result = agent.run(
        "Compute one design-check metric and retain the observation history.",
        request_id=request_id,
    )
    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "agents/basic/multi_step_json_with_memory.py",
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "steps_executed": output.get("steps_executed"),
        "memory_items": (
            len(output.get("memory", [])) if isinstance(output.get("memory"), list) else 0
        ),
        "tool_results_count": len(result.tool_results),
        "final_output": output.get("final_output"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)
    store.close()


if __name__ == "__main__":
    main()
