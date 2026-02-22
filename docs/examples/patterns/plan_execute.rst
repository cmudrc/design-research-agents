Plan Execute
============

Source: ``examples/patterns/plan_execute.py``

Introduction
------------

Plan-and-Solve and ReAct both separate planning from execution to reduce reasoning drift, while AutoGen
shows how these roles can be modularized across components. This example encodes planner-executor separation
with tool-backed execution and deterministic trace artifacts.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``PlannerExecutorPattern.run(...)`` with a fixed
   ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["PlannerExecutorPattern.run(...)"]
       C --> D["Planner and executor phases share tool/runtime state"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/plan_execute.py
   :language: python
   :lines: 61-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/plan_execute.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "patterns/plan_execute.py",
     "final_output": {
       "column_count": 2,
       "csv_path": "artifacts/examples/plan_execute_runtime_inventory.csv",
       "row_count": 3,
       "search_hits": 4
     },
     "plan_step_count": 1,
     "steps_executed": 1,
     "success": true,
     "terminated_reason": "completed",
     "trace": {
       "request_id": "example-workflow-plan-execute-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-workflow-plan-execute-design-001.jsonl"
     }
   }

References
----------

- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
