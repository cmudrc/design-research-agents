Workflow Model Step Design Tradeoff
===================================

Source: ``examples/workflow/workflow_model_step_design_tradeoff.py``

Introduction
------------

FrugalGPT frames cost-aware model choice, HELM frames robust comparative evaluation, and Toward Engineering
AGI frames engineering-task relevance of those choices. This example demonstrates model-step tradeoff
handling inside a workflow graph with deterministic trace capture.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["Workflow.run(...)"]
       C --> D["WorkflowRuntime schedules step graph (LogicStep, ModelStep)"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/workflow/workflow_model_step_design_tradeoff.py
   :language: python
   :lines: 59-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/workflow_model_step_design_tradeoff.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "workflow/workflow_model_step_design_tradeoff.py",
     "execution_order": [
       "design_tradeoff_model",
       "finalize"
     ],
     "final_output": {
       "tradeoff": "Use a modular latch for faster maintenance; accept small cost increase for serviceability."
     },
     "success": true,
     "terminated_reason": null,
     "trace": {
       "request_id": "example-workflow-model-step-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-model-step-design-001.jsonl"
     }
   }

References
----------

- `FrugalGPT <https://arxiv.org/abs/2305.05176>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
- `Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs <https://arxiv.org/abs/2509.16204>`_
