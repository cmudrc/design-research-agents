Remote
======

Source: ``examples/model_selection/remote.py``

Introduction
------------

Remote model selection has the same routing tradeoffs as local selection but adds external service
variability; FrugalGPT and RouteLLM motivate policy-driven routing, and Toward Engineering AGI motivates
engineering-task-aware evaluation of those routes. This example implements remote selection with
deterministic logging.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``ModelSelector.select(...)`` with a fixed ``request_id``.
3. Evaluate model constraints and policy, then expose selector metadata in the traced payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["ModelSelector.select(...)"]
       C --> D["policy and constraints resolve one model-selection outcome"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/model_selection/remote.py
   :language: python
   :lines: 59-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/model_selection/remote.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "catalog_signature": "4dbd48aeadb6",
     "example": "model_selection/remote.py",
     "model_id": "gpt-4o-mini",
     "policy_id": "default",
     "provider": "openai",
     "rationale": "priority=speed; selection_reason=high_load_remote; ram_budget_gb=10.0; max_latency_ms=800; gpu_p...
     "safety_constraints": {
       "max_cost_usd": null,
       "max_latency_ms": 800
     },
     "trace": {
       "request_id": "example-model-selection-remote-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162207Z_example-model-selection-remote-design-001.jsonl"
     }
   }

References
----------

- `FrugalGPT <https://arxiv.org/abs/2305.05176>`_
- `RouteLLM <https://arxiv.org/abs/2406.18665>`_
- `Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs <https://arxiv.org/abs/2509.16204>`_
