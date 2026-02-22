Multi Step JSON With Memory
===========================

Source: ``examples/agents/multi_step_json_with_memory.py``

Introduction
------------

Reflexion, Generative Agents, and MemGPT each emphasize that iterative performance improves when prior state
is persisted and reused rather than recomputed from scratch. This example adds memory reads/writes to JSON
tool-calling so multi-step behavior remains auditable across turns.

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
       C --> D["WorkflowRuntime loop enforces continuation and max-step policy"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/agents/multi_step_json_with_memory.py
   :language: python
   :lines: 62-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/multi_step_json_with_memory.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "agents/multi_step_json_with_memory.py",
     "final_output": {
       "char_count": 21,
       "line_count": 1,
       "unique_word_count": 3,
       "word_count": 3
     },
     "memory_items": 5,
     "steps_executed": 1,
     "success": true,
     "terminated_reason": "continuation_stopped:model",
     "tool_results_count": 1,
     "trace": {
       "request_id": "example-multi-step-json-memory-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-multi-step-json-memory-design-001.jsonl"
     }
   }

References
----------

- `Reflexion <https://arxiv.org/abs/2303.11366>`_
- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
