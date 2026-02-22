r"""# Patterns / RAG Reasoning.

## Introduction
RAG establishes retrieval-grounded generation, and memory-centric agent systems such as Generative Agents
and MemGPT show why persistent context is essential for longer design tasks. This example combines retrieval
and reasoning steps so grounded evidence flow is explicit in traces and outputs.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``RagReasoningPattern.run(...)`` with a fixed
   ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Persist and query context via ``SQLiteMemoryStore`` to demonstrate memory-backed workflow behavior.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["RagReasoningPattern.run(...)"]
    C --> D["retrieval and reasoning are composed via memory steps"]
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
     "example": "patterns/rag_reasoning.py",
     "final_output": {
       "artifacts": [],
       "final_output": "Prioritize maintainability checks and explicit testability criteria in the recommended arch...
       "model": "example-model",
       "model_text": "<truncated for docs>",
       "workflow": {
         "execution_order": [
           "prepare_request",
           "call_model",
           "finalize"
         ],
         "step_results": {
           "call_model": {
             "artifacts": [],
             "error": null,
             "metadata": {
               "stage": "execution"
             },
             "output": {
               "llm_response": {
                 "finish_reason": null,
                 "latency_ms": null,
                 "model": "example-model",
                 "provenance": null,
                 "provider": "example-test-monkeypatch",
                 "raw": null,
                 "raw_output": null,
                 "text": "<truncated for docs>",
                 "tool_calls": [],
                 "usage": null
               }
             },
             "status": "completed",
             "step_id": "call_model",
             "success": true
           },
           "finalize": {
             "artifacts": [],
             "error": null,
             "metadata": {
               "stage": "execution"
             },
             "output": {
               "metadata": {
                 "dependency_keys": [],
                 "llm_call": {
                   "max_tokens": null,
                   "message_count": 1,
                   "message_source": "prompt",
                   "provider_options_keys": [],
                   "response_schema_supplied": false,
                   "source": "direct",
                   "temperature": null
                 },
                 "request_id": "example-workflow-rag-design-001:rag_reasoning:workflow:reason"
               },
               "model_response": {
                 "finish_reason": null,
                 "latency_ms": null,
                 "model": "example-model",
                 "provenance": null,
                 "provider": "example-test-monkeypatch",
                 "raw": null,
                 "raw_output": null,
                 "text": "<truncated for docs>",
                 "tool_calls": [],
                 "usage": null
               },
               "output": {
                 "model": "example-model",
                 "model_text": "<truncated for docs>"
               }
             },
             "status": "completed",
             "step_id": "finalize",
             "success": true
           },
           "prepare_request": {
             "artifacts": [],
             "error": null,
             "metadata": {
               "stage": "execution"
             },
             "output": {
               "llm_request": {
                 "max_tokens": null,
                 "messages": [
                   {
                     "content": "Task: Draft a concise architecture recommendation for a serviceable edge device.

Retrieved context (JSON):
{
  "count": 1,
  "matches": [
    {
      "content": "Design requirement: include graceful shutdown and runtime monitoring.",
      "created_at": "2026-02-22T16:22:09.467704+00:00",
      "item_id": "7a77af51c8bf4871afe19f6a8a146ae4",
      "lexical_score": 0.0,
      "metadata": {
        "kind": "requirement"
      },
      "namespace": "design_examples",
      "score": 0.0,
      "updated_at": "2026-02-22T16:22:09.467704+00:00",
      "vector_score": null
    }
  ],
  "namespace": "design_examples"
}

Retrieved context (text):
- [7a77af51c8bf4871afe19f6a8a146ae4] score=0.0 Design requirement: include graceful shutdown and runtime monitoring.

Use the retrieved context when relevant, but reason independently when context is sparse.",
                     "name": null,
                     "role": "user",
                     "tool_call_id": null,
                     "tool_name": null
                   }
                 ],
                 "metadata": {
                   "agent": "DirectLLMCall",
                   "message_source": "prompt",
                   "request_id": "example-workflow-rag-design-001:rag_reasoning:workflow:reason"
                 },
                 "model": "example-model",
                 "provider_options": {},
                 "response_format": null,
                 "response_schema": null,
                 "task_profile": null,
                 "temperature": null,
                 "tools": []
               },
               "message_count": 1,
               "message_source": "prompt",
               "messages": [
                 {
                   "content": "Task: Draft a concise architecture recommendation for a serviceable edge device.

Retrieved context (JSON):
{
  "count": 1,
  "matches": [
    {
      "content": "Design requirement: include graceful shutdown and runtime monitoring.",
      "created_at": "2026-02-22T16:22:09.467704+00:00",
      "item_id": "7a77af51c8bf4871afe19f6a8a146ae4",
      "lexical_score": 0.0,
      "metadata": {
        "kind": "requirement"
      },
      "namespace": "design_examples",
      "score": 0.0,
      "updated_at": "2026-02-22T16:22:09.467704+00:00",
      "vector_score": null
    }
  ],
  "namespace": "design_examples"
}

Retrieved context (text):
- [7a77af51c8bf4871afe19f6a8a146ae4] score=0.0 Design requirement: include graceful shutdown and runtime monitoring.

Use the retrieved context when relevant, but reason independently when context is sparse.",
                   "name": null,
                   "role": "user",
                   "tool_call_id": null,
                   "tool_name": null
                 }
               ],
               "normalized_input": {
                 "prompt": "Task: Draft a concise architecture recommendation for a serviceable edge device.

Retrieved context (JSON):
{
  "count": 1,
  "matches": [
    {
      "content": "Design requirement: include graceful shutdown and runtime monitoring.",
      "created_at": "2026-02-22T16:22:09.467704+00:00",
      "item_id": "7a77af51c8bf4871afe19f6a8a146ae4",
      "lexical_score": 0.0,
      "metadata": {
        "kind": "requirement"
      },
      "namespace": "design_examples",
      "score": 0.0,
      "updated_at": "2026-02-22T16:22:09.467704+00:00",
      "vector_score": null
    }
  ],
  "namespace": "design_examples"
}

Retrieved context (text):
- [7a77af51c8bf4871afe19f6a8a146ae4] score=0.0 Design requirement: include graceful shutdown and runtime monitoring.

Use the retrieved context when relevant, but reason independently when context is sparse."
               },
               "resolved_model": "example-model"
             },
             "status": "completed",
             "step_id": "prepare_request",
             "success": true
           }
         },
         "success": true
       }
     },
     "success": true,
     "terminated_reason": "completed",
     "trace": {
       "request_id": "example-workflow-rag-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-workflow-rag-design-001.jsonl"
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
from design_research_agents.patterns import RagReasoningPattern


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
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        llm_client.close()
        store.close()

    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
