Rag
===

Source: ``examples/patterns/rag.py``

Introduction
------------

RAG establishes retrieval-grounded generation, and memory-centric agent systems such as Generative Agents
and MemGPT show why persistent context is essential for longer design tasks. This example combines retrieval
and reasoning steps so grounded evidence flow is explicit in traces and outputs.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``RAGPattern.run(...)`` with a fixed
   ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Persist and query context via ``SQLiteMemoryStore`` to demonstrate memory-backed workflow behavior.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["RAGPattern.run(...)"]
       C --> D["retrieval and reasoning are composed via memory steps"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/rag.py
   :language: python
   :lines: 53-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/patterns/rag.py

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

- `Retrieval-Augmented Generation <https://arxiv.org/abs/2005.11401>`_
- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
