"""Run traced ``RagReasoningPattern`` for design-memory reasoning.

Expected observations:
- retrieval reads seeded design memory records.
- final recommendation references retrieved context.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from pathlib import Path

from design_research_agents import RagReasoningPattern
from design_research_agents._contracts import MemoryWriteRecord
from design_research_agents._memory._stores._sqlite_store import SQLiteMemoryStore
from design_research_agents._shared._deterministic_design_helpers import EchoDesignReasoningAgent
from design_research_agents._shared._example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Run one local RAG workflow and print compact JSON result."""
    request_id = "example-workflow-rag-design-001"
    db_path = Path.cwd() / "artifacts" / "examples" / "rag_reasoning_example.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    store = SQLiteMemoryStore(db_path=db_path)
    store.write(
        [
            MemoryWriteRecord(
                content="Design requirement: include graceful shutdown and runtime monitoring.",
                metadata={"kind": "requirement"},
            )
        ],
        namespace="design_examples",
    )

    pattern = RagReasoningPattern(
        reasoning_delegate=EchoDesignReasoningAgent(),
        memory_store=store,
        memory_namespace="design_examples",
        memory_top_k=3,
        write_back=False,
        tracer=make_tracer(),
    )
    result = pattern.run(
        "Draft a concise architecture recommendation for a serviceable edge device.",
        request_id=request_id,
    )
    store.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "workflow/rag_reasoning.py",
        "success": result.success,
        "final_output": output.get("final_output"),
        "terminated_reason": output.get("terminated_reason"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
