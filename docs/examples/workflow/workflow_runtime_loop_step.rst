Workflow Runtime Loop Step
==========================

Source: ``examples/workflow/workflow_runtime_loop_step.py``

Introduction
------------

Tree of Thoughts and ReAct each motivate iterative reasoning with explicit state updates, and AutoGen
provides a practical framing for orchestrating repeated loop actions. This example demonstrates loop-step
execution in the workflow runtime, including bounded iteration behavior and trace emission.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

The diagram below is generated from the example's configured ``Workflow``.

.. mermaid::

   flowchart LR
       workflow_entry["Workflow Entrypoint"]
       step_1["design_counter_loop<br/>LoopStep<br/>max_iterations=10"]
       subgraph loop_body_1["Loop Body: design_counter_loop"]
           direction TD
           loop_entry_1["design_counter_loop iteration entry"]
           step_2["design_counter_loop::increment<br/>LogicStep"]
           step_3["design_counter_loop::snapshot<br/>LogicStep"]
           loop_entry_1 --> step_2
           step_2 --> step_3
           step_3 -. "next iteration" .-> loop_entry_1
       end
       workflow_entry --> step_1
       step_1 -. "iterate" .-> loop_entry_1

.. literalinclude:: ../../../examples/workflow/workflow_runtime_loop_step.py
   :language: python
   :lines: 40-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/workflow/workflow_runtime_loop_step.py

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

References
----------

- `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
