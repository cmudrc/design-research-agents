Ralph Loop
==========

Source: ``examples/patterns/ralph_loop.py``

Introduction
------------

Nominal-team style deliberation motivates structured role handoffs where proposal, critique, and evaluation
signals stay explicit. This example demonstrates a dynamic-role Ralph loop with evaluator-threshold stopping
and deterministic output traces.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build role delegates that emit deterministic proposal/critique/evaluation payloads.
3. Execute ``RalphLoopPattern.run(...)`` with typed ``LoopConfig`` and fixed ``request_id``.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["RalphLoopPattern.run(...)"]
       C --> D["role batch executes proposer/critic/evaluator each round"]
       C --> E["evaluator score compared to consensus threshold"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/ralph_loop.py
   :language: python
   :lines: 51-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/ralph_loop.py

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

- `Nominal Group Technique <https://doi.org/10.1177/104649648300400202>`_
- `Self-Refine <https://arxiv.org/abs/2303.17651>`_
- `Reflexion <https://arxiv.org/abs/2303.11366>`_
