r"""# Agents / Multi Step JSON With Memory.

## Introduction
Reflexion, Generative Agents, and MemGPT each emphasize that iterative performance improves when prior state
is persisted and reused rather than recomputed from scratch. This example adds memory reads/writes to JSON
tool-calling so multi-step behavior remains auditable across turns.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Persist and query context via ``SQLiteMemoryStore`` to demonstrate memory-backed workflow behavior.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["MultiStepAgent.run(...)"]
    C --> D["WorkflowRuntime loop enforces continuation and max-step policy"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "agents/multi_step_json_with_memory.py",
     "final_output": {
       "char_count": 21,
       "line_count": 1,
       "unique_word_count": 3,
       "word_count": 3
     },
     "memory_items": 5,
     "steps_executed": 1,
     "success": true,
     "terminated_reason": "continuation_stopped:model",
     "tool_results_count": 1,
     "trace": {
       "request_id": "example-multi-step-json-memory-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-multi-step-json-memory-design-001.jsonl"
     }
   }


## References
- `Reflexion <https://arxiv.org/abs/2303.11366>`_
- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Toolbox, Tracer
from design_research_agents.memory import SQLiteMemoryStore


def main() -> None:
    """Run one multi-step JSON tool call with memory retrieval and write-back."""
    # Keep the request id stable so trace filenames and test snapshots stay comparable.
    request_id = "example-multi-step-json-memory-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    db_path = Path("artifacts/examples/multi_step_json_with_memory.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Recreate the DB per run to keep the example deterministic across repeated executions.
    if db_path.exists():
        db_path.unlink()

    tool_runtime = Toolbox()

    # Seed one memory item so the agent can demonstrate retrieval-conditioned behavior.
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
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        # Explicit shutdown keeps local handles/sockets from leaking in long-lived sessions.
        llm_client.close()
        tool_runtime.close()
        store.close()
    memory_items = result.output_list("memory")
    summary = result.summary(
        details={"memory_items": len(memory_items)},
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
