Local
=====

Source: ``examples/model_selection/local.py``

Introduction
------------

FrugalGPT and RouteLLM both frame model selection as a policy problem balancing capability, latency, and
cost, while HELM stresses evaluation rigor across model choices. This example demonstrates local model
selection policy execution with observable scoring outputs and traces.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build a default ``ModelFlightRegistry`` and flatten it into the ``ModelCatalog`` passed to ``ModelSelector``.
3. Execute ``ModelSelector.select(...)`` with a fixed ``request_id``.
4. Evaluate model constraints and policy, then expose selector metadata in the traced payload.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["ModelSelector.select(...)"]
       C --> D["policy and constraints resolve one model-selection outcome"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/model_selection/local.py
   :language: python
   :lines: 59-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/model_selection/local.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "catalog_signature": "4dbd48aeadb6",
     "example": "model_selection/local.py",
     "model_id": "llama-3.1-8b-instruct-gguf-q4_k_m",
     "policy_id": "default",
     "provider": "llama_cpp",
     "rationale": "priority=quality; selection_reason=local_fit; model_size_b=8.0; ram_budget_gb=10.0; max_cost_us...
     "safety_constraints": {
       "max_cost_usd": 0.01,
       "max_latency_ms": null
     },
     "trace": {
       "request_id": "example-model-selection-local-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162207Z_example-model-selection-local-design-001.jsonl"
     }
   }

References
----------

- `FrugalGPT <https://arxiv.org/abs/2305.05176>`_
- `RouteLLM <https://arxiv.org/abs/2406.18665>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
