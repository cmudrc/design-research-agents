SGLang Server Client
====================

Source: ``examples/clients/sglang_server_client.py``

Introduction
------------

SGLang focuses on high-throughput serving and exposes OpenAI-compatible APIs, making it useful for
controlled backend substitution against common response contracts and HELM-style evaluation framing. This
example wires the SGLang server client into the same traced run surface used by other providers.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``SGLangServerLLMClient.generate(...)`` with a
   fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["SGLangServerLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/clients/sglang_server_client.py
   :language: python
   :lines: 85-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src DRA_EXAMPLE_LLM_MODE=deterministic python examples/clients/sglang_server_client.py

This checkout-only command reproduces the documented output without a
live backend. For real installs, credentials, and backend-specific setup,
see :doc:`../../llm_clients/index`.

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "base_url": "http://127.0.0.1:30000/v1",
       "default_model": "Qwen/Qwen2.5-1.5B-Instruct",
       "host": "127.0.0.1",
       "kind": "sglang_server",
       "max_retries": 3,
       "model_patterns": [
         "Qwen/*",
         "qwen2.5-*"
       ],
       "name": "sglang-local-dev",
       "port": 30000
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "SGLangServerLLMClient",
     "default_model": "Qwen/Qwen2.5-1.5B-Instruct",
     "example": "clients/sglang_server_client.py",
     "llm_call": {
       "prompt": "Provide one sentence on when SGLang-style serving helps local benchmarking.",
       "response_has_text": true,
       "response_model": "Qwen/Qwen2.5-1.5B-Instruct",
       "response_provider": "example-test-monkeypatch",
       "response_text": "SGLang-style serving helps when you need stable local throughput for repeated tests."
     },
     "server": {
       "host": "127.0.0.1",
       "kind": "sglang_server",
       "managed": true,
       "port": 30000
     },
     "trace": {
       "request_id": "example-clients-sglang-server-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-sglang-server-call-001.jsonl"
     }
   }

References
----------

- `SGLang OpenAI-Compatible API <https://docs.sglang.ai/basic_usage/openai_api.html>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
