Tree Search
===========

Source: ``examples/patterns/tree_search.py``

Introduction
------------

Tree of Thoughts motivates explicit branching and ranking instead of single-pass revision.
This example uses dedicated generator/evaluator delegates and a bounded beam search to show
search-policy behavior (expand, score, prune) in a traceable way.

.. note::

   This example's checked-in local ``LlamaCppServerLLMClient`` config uses a
   ``Qwen3-4B`` GGUF model. On lower-RAM machines, swap in a smaller local
   model or start with :doc:`../clients/ollama_local_client`.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build generator and evaluator delegates with ``DirectLLMCall`` and a managed ``LlamaCppServerLLMClient``.
3. Execute ``TreeSearchPattern.run(...)`` with explicit search controls and preserve frontier diagnostics.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["TreeSearchPattern.run(...)"]
       C --> D["generator/evaluator delegates expand and score candidate nodes"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/tree_search.py
   :language: python
   :lines: 57-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/patterns/tree_search.py

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
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
