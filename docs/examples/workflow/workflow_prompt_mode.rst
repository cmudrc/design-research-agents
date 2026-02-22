Workflow Prompt Mode
====================

Source: ``examples/workflow/workflow_prompt_mode.py``

Introduction
------------

ReAct and Plan-and-Solve motivate explicit control over reasoning phases, and JSON Schema formalizes
structured inputs/outputs when prompt-mode steps need predictable contracts. This example shows prompt-mode
workflow composition with agent, logic, and tool steps under one runtime.

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
       C --> D["WorkflowRuntime schedules step graph (AgentStep, LogicStep, ToolStep)"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/workflow/workflow_prompt_mode.py
   :language: python
   :lines: 91-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/workflow_prompt_mode.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "agent_branch_run": {
       "error": null,
       "example": "workflow/workflow_prompt_mode.py",
       "execution_order": [
         "router",
         "draft_agent",
         "parse_agent_json",
         "finalize_agent",
         "draft_template",
         "finalize_template"
       ],
       "final_output": {
         "branch": "agent",
         "summary": "Use one runtime that fuses core, script, and MCP tools.",
         "title": "Deterministic workflow memo"
       },
       "success": true,
       "terminated_reason": null,
       "trace": {
         "request_id": "example-workflow-prompt-design-agent-001",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-prompt-design-agent-001.jsonl"
       }
     },
     "template_branch_run": {
       "error": null,
       "example": "workflow/workflow_prompt_mode.py",
       "execution_order": [
         "router",
         "draft_agent",
         "parse_agent_json",
         "finalize_agent",
         "draft_template",
         "finalize_template"
       ],
       "final_output": {
         "branch": "template",
         "summary": "Template mode output for: template: Produce a deterministic fallback brief for manufacturabili...
         "title": "Template fallback design brief"
       },
       "success": true,
       "terminated_reason": null,
       "trace": {
         "request_id": "example-workflow-prompt-design-template-001",
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
