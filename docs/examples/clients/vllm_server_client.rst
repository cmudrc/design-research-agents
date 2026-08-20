vLLM Server Client
==================

Source: ``examples/clients/vllm_server_client.py``

Introduction
------------

vLLM is a common high-performance inference server, OpenAI-compatible response contracts enable drop-in
orchestration reuse, and HELM provides context for why consistent serving interfaces help evaluation. This
example exercises the vLLM server client integration with explicit trace reporting.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``VLLMServerLLMClient.generate(...)`` with a
   fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["VLLMServerLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/clients/vllm_server_client.py
   :language: python
   :lines: 84-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src DRA_EXAMPLE_LLM_MODE=deterministic python examples/clients/vllm_server_client.py

This checkout-only command reproduces the documented output without a
live backend. For real installs, credentials, and backend-specific setup,
see :doc:`../../llm_clients/index`.

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "base_url": "http://127.0.0.1:8002/v1",
       "default_model": "qwen2.5-1.5b-instruct",
       "host": "127.0.0.1",
       "kind": "vllm_server",
       "max_retries": 3,
       "model_patterns": [
         "qwen2.5-*"
       ],
       "name": "vllm-local-dev",
       "port": 8002
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "VLLMServerLLMClient",
     "default_model": "qwen2.5-1.5b-instruct",
     "example": "clients/vllm_server_client.py",
     "llm_call": {
       "prompt": "Provide one sentence on why local serving helps reproducible benchmarking.",
       "response_has_text": true,
       "response_model": "qwen2.5-1.5b-instruct",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Local serving reduces backend drift and improves benchmark reproducibility."
     },
     "server": {
       "host": "127.0.0.1",
       "kind": "vllm_server",
       "managed": true,
       "port": 8002
     },
     "trace": {
       "request_id": "example-clients-vllm-server-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-vllm-server-call-001.jsonl"
     }
   }

References
----------

- `vLLM OpenAI-Compatible Server <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
