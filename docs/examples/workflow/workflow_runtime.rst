Workflow Runtime
================

Source: ``examples/workflow/workflow_runtime.py``

Introduction
------------

Human-AI collaboration by design motivates transparent orchestration boundaries, AutoGen motivates
composable multi-component execution, and HELM motivates repeatable runtime instrumentation for comparisons.
This example is the minimal workflow-runtime build for observing step execution semantics directly.

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
       step_1["design_runtime_ready<br/>LogicStep"]
       workflow_entry --> step_1

.. literalinclude:: ../../../examples/workflow/workflow_runtime.py
   :language: python
   :lines: 40-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/workflow/workflow_runtime.py

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

- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
