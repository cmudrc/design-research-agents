OpenAI Compatible HTTP Client
=============================

Source: ``examples/clients/openai_compatible_http_client.py``

Introduction
------------

OpenAI-compatible HTTP surfaces are valuable because they let one orchestration stack target multiple
providers; vLLM and SGLang both expose this style of interface while OpenAI Responses API defines the
baseline semantics. This example demonstrates that compatibility layer in the framework client runtime.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``OpenAICompatibleHTTPLLMClient.generate(...)``
   with a fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["OpenAICompatibleHTTPLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/clients/openai_compatible_http_client.py
   :language: python
   :lines: 79-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src DRA_EXAMPLE_LLM_MODE=deterministic python examples/clients/openai_compatible_http_client.py

This checkout-only command reproduces the documented output without a
live backend. For real installs, credentials, and backend-specific setup,
see :doc:`../../llm_clients/index`.

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "api_key_env": "OPENAI_API_KEY",
       "base_url": "http://127.0.0.1:8011/v1",
       "default_model": "qwen2.5-1.5b-q4",
       "kind": "openai_compatible_http",
       "max_retries": 3,
       "model_patterns": [
         "qwen2.5-*",
         "qwen2-*"
       ],
       "name": "local-openai-compat"
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "OpenAICompatibleHTTPLLMClient",
     "default_model": "qwen2.5-1.5b-q4",
     "example": "clients/openai_compatible_http_client.py",
     "llm_call": {
       "prompt": "Provide one sentence on balancing latency and quality in design review assistants.",
       "response_has_text": true,
       "response_model": "qwen2.5-1.5b-q4",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Use fast drafts for iteration, then escalate critical decisions to higher-quality models."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-openai-compatible-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-openai-compatible-call-001.jsonl"
     }
   }

References
----------

- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `vLLM OpenAI-Compatible Server <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>`_
- `SGLang OpenAI-Compatible API <https://docs.sglang.ai/basic_usage/openai_api.html>`_
