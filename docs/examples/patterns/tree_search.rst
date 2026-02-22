Tree Search
===========

Source: ``examples/patterns/tree_search.py``

Introduction
------------

Tree of Thoughts motivates branching deliberation over single-chain prompting, while Plan-and-Solve and
ReAct provide complementary stepwise control principles. This example instantiates tree-search reasoning as
an inspectable pattern for comparing branch quality under fixed runtime controls.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``TreeSearchPattern.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["TreeSearchPattern.run(...)"]
       C --> D["generator/evaluator loop expands and prunes candidate tree"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/tree_search.py
   :language: python
   :lines: 63-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/tree_search.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "best_candidate": {
       "concept": "modular frame + fail-safe",
       "score_hint": 0.92
     },
     "error": null,
     "example": "patterns/tree_search.py",
     "final_output": {
       "best_candidate": {
         "concept": "modular frame + fail-safe",
         "score_hint": 0.92
       },
       "best_score": 0.92
     },
     "success": true,
     "terminated_reason": "max_depth_reached",
     "trace": {
       "request_id": "example-workflow-tree-search-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-workflow-tree-search-design-001.jsonl"
     }
   }

References
----------

- `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
