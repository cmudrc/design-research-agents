r"""# Workflow / Workflow Delegate And Memory Steps.

## Introduction
Generative Agents and MemGPT both emphasize durable memory as a first-class runtime primitive, while AutoGen
demonstrates delegation across specialized roles. This example composes delegate and memory steps in a
single workflow so context propagation and role handoff remain explicit.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Persist and query context via ``SQLiteMemoryStore`` to demonstrate memory-backed workflow behavior.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["Workflow.run(...)"]
    C --> D["WorkflowRuntime schedules step graph (DelegateBatchStep, LogicStep, MemoryReadStep, MemoryWriteStep)"]
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
     "example": "workflow/workflow_delegate_and_memory_steps.py",
     "execution_order": [
       "seed_constraints",
       "read_constraints",
       "peer_batch",
       "finalize"
     ],
     "final_output": {
       "constraints_found": 2,
       "delegate_calls": 2,
       "final_delegate_output": {
         "artifacts": [],
         "final_output": "Add gasket alignment features to preserve ingress protection after service.",
         "model": "example-model",
         "model_text": "Add gasket alignment features to preserve ingress protection after service.",
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
                   "text": "Add gasket alignment features to preserve ingress protection after service.",
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
                   "request_id": "<truncated-request-id>"
                 },
                 "model_response": {
                   "finish_reason": null,
                   "latency_ms": null,
                   "model": "example-model",
                   "provenance": null,
                   "provider": "example-test-monkeypatch",
                   "raw": null,
                   "raw_output": null,
                   "text": "Add gasket alignment features to preserve ingress protection after service.",
                   "tool_calls": [],
                   "usage": null
                 },
                 "output": {
                   "model": "example-model",
                   "model_text": "Add gasket alignment features to preserve ingress protection after service."
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
                       "content": "Propose reliability-focused maintenance improvements.",
                       "name": null,
                       "role": "user",
                       "tool_call_id": null,
                       "tool_name": null
                     }
                   ],
                   "metadata": {
                     "agent": "DirectLLMCall",
                     "message_source": "prompt",
                     "request_id": "<truncated-request-id>"
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
                     "content": "Propose reliability-focused maintenance improvements.",
                     "name": null,
                     "role": "user",
                     "tool_call_id": null,
                     "tool_name": null
                   }
                 ],
                 "normalized_input": {
                   "prompt": "Propose reliability-focused maintenance improvements."
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
       }
     },
     "success": true,
     "terminated_reason": null,
     "trace": {
       "request_id": "example-workflow-delegate-memory-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-delegate-memory-design-001.jsonl"
     }
   }


## References
- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DelegateBatchStep,
    DirectLLMCall,
    LlamaCppServerLLMClient,
    LogicStep,
    MemoryReadStep,
    MemoryWriteStep,
    Tracer,
    Workflow,
)
from design_research_agents.memory import SQLiteMemoryStore


def main() -> None:
    """Execute memory and delegate-batch primitives in one traced workflow."""
    # Stable request ids keep workflow trace artifacts deterministic for docs snapshots.
    request_id = "example-workflow-delegate-memory-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    db_path = Path("artifacts/examples/workflow_delegate_and_memory.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Reset persisted state so each example run starts from the same memory baseline.
    if db_path.exists():
        db_path.unlink()

    store = SQLiteMemoryStore(db_path=db_path)
    llm_client = LlamaCppServerLLMClient()
    try:
        # Two delegates share the same backend client to model role-specific prompts over one transport.
        manufacturing_peer = DirectLLMCall(llm_client=llm_client, tracer=tracer)
        reliability_peer = DirectLLMCall(llm_client=llm_client, tracer=tracer)

        workflow = Workflow(
            tool_runtime=None,
            memory_store=store,
            tracer=tracer,
            input_schema={"type": "object"},
            steps=[
                MemoryWriteStep(
                    step_id="seed_constraints",
                    namespace="design_constraints",
                    # Seed memory first so downstream reads/delegates operate on explicit constraints.
                    records_builder=lambda _context: [
                        {
                            "content": "Constraint: reduce service time by at least 20 percent.",
                            "metadata": {"kind": "constraint"},
                        },
                        {
                            "content": "Constraint: preserve ingress protection sealing.",
                            "metadata": {"kind": "constraint"},
                        },
                    ],
                ),
                MemoryReadStep(
                    step_id="read_constraints",
                    namespace="design_constraints",
                    dependencies=("seed_constraints",),
                    top_k=5,
                    query_builder=lambda _context: {
                        "text": "service time constraint",
                        "metadata_filters": {"kind": "constraint"},
                    },
                ),
                DelegateBatchStep(
                    step_id="peer_batch",
                    dependencies=("read_constraints",),
                    fail_fast=False,
                    calls_builder=lambda context: [
                        {
                            "call_id": "manufacturing_peer",
                            "delegate": manufacturing_peer,
                            "prompt": (
                                "Propose manufacturing-friendly maintenance improvements using "
                                "retrieved constraints count="
                                f"{context['dependency_results']['read_constraints']['output']['count']}."
                            ),
                        },
                        {
                            "call_id": "reliability_peer",
                            "delegate": reliability_peer,
                            "prompt": "Propose reliability-focused maintenance improvements.",
                        },
                    ],
                ),
                LogicStep(
                    step_id="finalize",
                    dependencies=("read_constraints", "peer_batch"),
                    handler=lambda context: {
                        "constraints_found": (context["dependency_results"]["read_constraints"]["output"]["count"]),
                        "delegate_calls": len(context["dependency_results"]["peer_batch"]["output"].get("results", [])),
                        "final_delegate_output": (
                            context["dependency_results"]["peer_batch"]["output"].get("final_output")
                        ),
                    },
                ),
            ],
        )

        result = workflow.run({}, request_id=request_id)
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        llm_client.close()
        store.close()

    summary = result.summary(
        details={"execution_order": list(result.execution_order)},
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
