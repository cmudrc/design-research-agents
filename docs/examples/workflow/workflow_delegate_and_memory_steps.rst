Workflow Delegate And Memory Steps
==================================

Source: ``examples/workflow/workflow_delegate_and_memory_steps.py``

Introduction
------------

Generative Agents and MemGPT both emphasize durable memory as a first-class runtime primitive, while AutoGen
demonstrates delegation across specialized roles. This example composes delegate and memory steps in a
single workflow so context propagation and role handoff remain explicit.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Persist and query context via ``SQLiteMemoryStore`` to demonstrate memory-backed workflow behavior.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["Workflow.run(...)"]
       C --> D["WorkflowRuntime schedules step graph (DelegateBatchStep, LogicStep, MemoryReadStep, MemoryWriteStep)"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/workflow/workflow_delegate_and_memory_steps.py
   :language: python
   :lines: 195-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/workflow_delegate_and_memory_steps.py

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

References
----------

- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
