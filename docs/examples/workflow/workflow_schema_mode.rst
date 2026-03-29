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

The diagram below is generated from the example's configured ``Workflow``.

.. mermaid::

   flowchart LR
       workflow_entry["Workflow Entrypoint"]
       step_1["describe_dataset<br/>ToolStep<br/>tool=data.describe"]
       step_2["load_sample<br/>ToolStep<br/>tool=data.load_csv"]
       step_3["quality_gate<br/>LogicStep"]
       step_4["persist_report<br/>ToolStep<br/>tool=fs.write_text"]
       step_5["finalize<br/>LogicStep"]
       workflow_entry --> step_1
       step_1 --> step_2
       step_1 --> step_3
       step_2 --> step_3
       step_3 --> step_4
       step_4 --> step_5

.. literalinclude:: ../../../examples/workflow/workflow_schema_mode.py
   :language: python
   :lines: 53-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/workflow_schema_mode.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "strict_run": {
       "success": true,
       "final_output": "<example-specific payload>",
       "terminated_reason": "<string-or-null>",
       "error": null,
       "trace": {
         "request_id": "<request-id>",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     },
     "relaxed_run": {
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
   }

References
----------

- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
