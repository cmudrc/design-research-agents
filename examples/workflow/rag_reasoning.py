"""Run traced ``RagReasoningPattern`` for design-memory reasoning.

Expected observations:
- retrieval reads seeded design memory records.
- final recommendation references retrieved context.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from _support_deterministic import EchoDesignReasoningAgent

from design_research_agents import RagReasoningPattern, Toolbox, Tracer
from design_research_agents.memory import SQLiteMemoryStore


def main() -> None:
    """Run one local RAG workflow and print compact JSON result."""
    request_id = "example-workflow-rag-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    db_path = Path.cwd() / "artifacts" / "examples" / "rag_reasoning_example.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    seed_toolbox = Toolbox()
    seed_toolbox.invoke_dict(
        "memory.write",
        {
            "db_path": str(db_path),
            "namespace": "design_examples",
            "records": [
                {
                    "content": "Design requirement: include graceful shutdown and runtime monitoring.",
                    "metadata": {"kind": "requirement"},
                }
            ],
        },
        request_id=f"{request_id}:seed_memory",
        dependencies={},
    )
    seed_toolbox.close()

    store = SQLiteMemoryStore(db_path=db_path)

    pattern = RagReasoningPattern(
        reasoning_delegate=EchoDesignReasoningAgent(),
        memory_store=store,
        memory_namespace="design_examples",
        memory_top_k=3,
        write_back=False,
        tracer=tracer,
    )
    result = pattern.run(
        "Draft a concise architecture recommendation for a serviceable edge device.",
        request_id=request_id,
    )
    store.close()

    payload = {
        "example": "workflow/rag_reasoning.py",
        "success": result.success,
        "final_output": result.final_output,
        "terminated_reason": result.terminated_reason,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
