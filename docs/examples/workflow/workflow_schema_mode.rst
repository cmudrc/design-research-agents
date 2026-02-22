Workflow Schema Mode
====================

Source: ``examples/workflow/workflow_schema_mode.py``

Introduction
------------

JSON Schema and function-calling conventions are central for reliable machine-to-machine workflow steps,
while the Responses API anchors current structured request/response patterns. This example illustrates
schema-mode workflow execution where each step contract is explicit and testable.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["Workflow.run(...)"]
       C --> D["WorkflowRuntime schedules step graph (LogicStep, ToolStep)"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/workflow/workflow_schema_mode.py
   :language: python
   :lines: 85-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/workflow_schema_mode.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "relaxed_run": {
       "error": null,
       "example": "workflow/workflow_schema_mode.py",
       "execution_order": [
         "describe_dataset",
         "load_sample",
         "quality_gate",
         "persist_report",
         "finalize"
       ],
       "final_output": {
         "report_path": "artifacts/examples/<truncated-report-path>"
       },
       "success": true,
       "terminated_reason": null,
       "trace": {
         "request_id": "example-workflow-schema-design-relaxed-001",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-schema-design-relaxed-001.jsonl"
       }
     },
     "strict_run": {
       "error": null,
       "example": "workflow/workflow_schema_mode.py",
       "execution_order": [
         "describe_dataset",
         "load_sample",
         "quality_gate",
         "persist_report",
         "finalize"
       ],
       "final_output": {
         "report_path": "artifacts/examples/<truncated-report-path>"
       },
       "success": true,
       "terminated_reason": null,
       "trace": {
         "request_id": "example-workflow-schema-design-strict-001",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-schema-design-strict-001.jsonl"
       }
     }
   }

References
----------

- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
