"""# Patterns / RAG.

## Introduction
RAG establishes retrieval-grounded generation, and memory-centric agent systems such as Generative Agents
and MemGPT show why persistent context is essential for longer design tasks. This example combines retrieval
and reasoning steps so grounded evidence flow is explicit in traces and outputs.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``RAGPattern.run(...)`` with a fixed
   ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Persist and query context via ``SQLiteMemoryStore`` to demonstrate memory-backed workflow behavior.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["RAGPattern.run(...)"]
    C --> D["retrieval and reasoning are composed via memory steps"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<example-specific payload>",
     "terminated_reason": "<string-or-null>",
     "error": null,
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }

## References
- `Retrieval-Augmented Generation <https://arxiv.org/abs/2005.11401>`_
- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient, Toolbox, Tracer
from design_research_agents.memory import SQLiteMemoryStore
from design_research_agents.patterns import RAGPattern


def main() -> None:
    """Run one local RAG workflow and print compact JSON result."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-workflow-rag-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    db_path = Path.cwd() / "artifacts" / "examples" / "rag_example.sqlite3"
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
    with LlamaCppServerLLMClient() as llm_client:
        pattern = RAGPattern(
            reasoning_delegate=DirectLLMCall(llm_client=llm_client, tracer=tracer),
            memory_store=store,
            memory_namespace="design_examples",
            memory_top_k=3,
            write_back=False,
            tracer=tracer,
        )
        try:
            result = pattern.run(
                "Draft a concise architecture recommendation for a serviceable edge device.",
                request_id=request_id,
            )
        finally:
            store.close()

    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
