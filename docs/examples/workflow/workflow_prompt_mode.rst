Workflow Prompt Mode
====================

Source: ``examples/workflow/workflow_prompt_mode.py``

Introduction
------------

ReAct and Plan-and-Solve motivate explicit control over reasoning phases, and JSON Schema formalizes
structured inputs/outputs when prompt-mode steps need predictable contracts. This example shows prompt-mode
workflow composition with agent, logic, and tool steps under one runtime, including one packaged-problem-like
object passed directly to ``Workflow.run(...)``.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``,
   once from a packaged-problem-like object and once from a plain fallback string prompt.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

The diagram below is generated from the example's configured ``Workflow``.

.. mermaid::

   flowchart LR
       workflow_entry["Workflow Entrypoint"]
       step_1["router<br/>LogicStep"]
       step_2["draft_agent<br/>DelegateStep<br/>delegate=_DocDelegate"]
       step_3["parse_agent_json<br/>ToolStep<br/>tool=text.extract_json"]
       step_4["finalize_agent<br/>LogicStep"]
       step_5["draft_template<br/>LogicStep"]
       step_6["finalize_template<br/>LogicStep"]
       workflow_entry --> step_1
       step_1 -. "route=agent_path" .-> step_2
       step_1 -. "route=template_path" .-> step_5
       step_2 --> step_3
       step_3 --> step_4
       step_5 --> step_6

.. literalinclude:: ../../../examples/workflow/workflow_prompt_mode.py
   :language: python
   :lines: 55-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/workflow/workflow_prompt_mode.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "agent_branch_run": {
       "success": true,
       "final_output": "<evaluation-ready final_output payload>",
       "terminated_reason": "<string-or-null>",
       "error": null,
       "trace": {
         "request_id": "<request-id>",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     },
     "template_branch_run": {
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

- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
