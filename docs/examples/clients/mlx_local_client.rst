MLX Local Client
================

Source: ``examples/clients/mlx_local_client.py``

Introduction
------------

MLX-LM provides an Apple-silicon-native local inference stack, HELM motivates reproducible evaluation
baselines, and AI-assisted design synthesis work connects these runtimes to educational design workflows.
This example exercises the MLX local client path with trace artifacts suitable for repeatable comparisons.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MlxLocalLLMClient.generate(...)`` with a fixed
   ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["MlxLocalLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/clients/mlx_local_client.py
   :language: python
   :lines: 80-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/clients/mlx_local_client.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "base_url": null,
       "default_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
       "kind": "mlx_local",
       "max_retries": 2,
       "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
       "model_patterns": [
         "mlx-community/*",
         "qwen2.5-*"
       ],
       "name": "mlx-local-dev",
       "quantization": "4bit"
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "MlxLocalLLMClient",
     "default_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
     "example": "clients/mlx_local_client.py",
     "llm_call": {
       "prompt": "Give one concise guideline for maintainable design telemetry schemas.",
       "response_has_text": true,
       "response_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Keep schema fields stable, documented, and versioned for comparability."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-mlx-local-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-mlx-local-call-001.jsonl"
     }
   }

References
----------

- `MLX-LM <https://github.com/ml-explore/mlx-lm>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
- `AI-assisted design synthesis and human creativity in engineering education <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1714523/full>`_
