Multi Step JSON With Memory
===========================

Source: ``examples/agents/multi_step_json_with_memory.py``

Introduction
------------

Reflexion, Generative Agents, and MemGPT each emphasize that iterative performance improves when prior state
is persisted and reused rather than recomputed from scratch. This example adds memory reads/writes to JSON
tool-calling so multi-step behavior remains auditable across turns.

.. note::

   This example's checked-in local ``LlamaCppServerLLMClient`` config uses a
   ``Qwen3-4B`` GGUF model. On lower-RAM machines, swap in a smaller local
   model or start with :doc:`../clients/ollama_local_client`.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Persist and query context via ``SQLiteMemoryStore`` to demonstrate memory-backed workflow behavior.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["MultiStepAgent.run(...)"]
       C --> D["WorkflowRuntime loop enforces explicit final-answer and max-step policy"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/agents/multi_step_json_with_memory.py
   :language: python
   :lines: 58-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/multi_step_json_with_memory.py

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

- `Reflexion <https://arxiv.org/abs/2303.11366>`_
- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
