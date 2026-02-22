"""Example script.

Motivation
Run traced ``RagReasoningPattern`` for design-memory reasoning.

Diagram
```mermaid
flowchart LR
    A["Pattern prompt"] --> B["Pattern orchestration"]
    B --> C["rag reasoning result"]
    C --> D["Trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `patterns` use-cases and run `rag_reasoning`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/patterns/rag_reasoning.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient, RagReasoningPattern, Toolbox, Tracer
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
    llm_client = LlamaCppServerLLMClient()
    try:
        pattern = RagReasoningPattern(
            reasoning_delegate=DirectLLMCall(llm_client=llm_client, tracer=tracer),
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
    finally:
        llm_client.close()
        store.close()

    payload = {
        "example": "patterns/rag_reasoning.py",
        "success": result.success,
        "final_output": result.final_output,
        "terminated_reason": result.terminated_reason,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
