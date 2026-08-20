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

The diagram below is generated from the example's configured ``Workflow``.

.. mermaid::

   flowchart LR
       workflow_entry["Workflow Entrypoint"]
       step_1["seed_constraints<br/>MemoryWriteStep<br/>namespace=design_constraints"]
       step_2["read_constraints<br/>MemoryReadStep<br/>namespace=design_constraints"]
       step_3["peer_batch<br/>DelegateBatchStep<br/>batch delegate calls"]
       step_4["finalize<br/>LogicStep"]
       workflow_entry --> step_1
       step_1 --> step_2
       step_2 --> step_3
       step_2 --> step_4
       step_3 --> step_4

.. literalinclude:: ../../../examples/workflow/workflow_delegate_and_memory_steps.py
   :language: python
   :lines: 41-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/workflow/workflow_delegate_and_memory_steps.py

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

- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
