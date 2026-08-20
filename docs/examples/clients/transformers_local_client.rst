Transformers Local Client
=========================

Source: ``examples/clients/transformers_local_client.py``

Introduction
------------

Transformers pipelines are often the first local baseline for experimentation, HELM stresses the value of
consistent evaluation scaffolding, and AI-assisted design education literature motivates reproducible local
setups for pedagogy. This example demonstrates the Transformers local client path with deterministic trace
output.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``TransformersLocalLLMClient.generate(...)``
   with a fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["TransformersLocalLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/clients/transformers_local_client.py
   :language: python
   :lines: 83-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src DRA_EXAMPLE_LLM_MODE=deterministic python examples/clients/transformers_local_client.py

This checkout-only command reproduces the documented output without a
live backend. For real installs, credentials, and backend-specific setup,
see :doc:`../../llm_clients/index`.

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "base_url": null,
       "default_model": "Qwen/Qwen2.5-1.5B-Instruct",
       "device": "auto",
       "dtype": "auto",
       "kind": "transformers_local",
       "max_retries": 2,
       "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
       "model_patterns": [
         "Qwen/*",
         "qwen2.5-*"
       ],
       "name": "transformers-local-dev",
       "quantization": "none"
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "TransformersLocalLLMClient",
     "default_model": "Qwen/Qwen2.5-1.5B-Instruct",
     "example": "clients/transformers_local_client.py",
     "llm_call": {
       "prompt": "Provide one sentence on why deterministic local runs aid design reproducibility.",
       "response_has_text": true,
       "response_model": "Qwen/Qwen2.5-1.5B-Instruct",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Deterministic local runs make design comparisons repeatable across experiments."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-transformers-local-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-transformers-local-call-001.jsonl"
     }
   }

References
----------

- `Transformers Pipeline Tutorial <https://huggingface.co/docs/transformers/main/en/pipeline_tutorial>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
- `AI-assisted design synthesis and human creativity in engineering education <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1714523/full>`_
