Llama CPP Server Client
=======================

Source: ``examples/clients/llama_cpp_server_client.py``

Introduction
------------

Local serving with llama.cpp is a practical path for controllable offline experimentation, OpenAI-style
response contracts improve interchangeability, and HELM motivates standardized evaluation conditions. This
example validates the llama.cpp server client path with tracing and deterministic output framing.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``LlamaCppServerLLMClient.generate(...)`` with a
   fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["LlamaCppServerLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/clients/llama_cpp_server_client.py
   :language: python
   :lines: 86-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src DRA_EXAMPLE_LLM_MODE=deterministic python examples/clients/llama_cpp_server_client.py

This checkout-only command reproduces the documented output without a
live backend. For real installs, credentials, and backend-specific setup,
see :doc:`../../llm_clients/index`.

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "api_model": "qwen2.5-1.5b-q4",
       "base_url": null,
       "default_model": "qwen2.5-1.5b-q4",
       "host": "127.0.0.1",
       "kind": "llama_cpp_server",
       "max_retries": 3,
       "model_patterns": [
         "qwen2.5-*",
         "qwen2-*"
       ],
       "name": "llama-local-dev",
       "port": 8011
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "LlamaCppServerLLMClient",
     "default_model": "qwen2.5-1.5b-q4",
     "example": "clients/llama_cpp_server_client.py",
     "llm_call": {
       "prompt": "In one sentence, explain a key tradeoff in engineering design reviews.",
       "response_has_text": true,
       "response_model": "qwen2.5-1.5b-q4",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Tradeoff: strict review gates improve reliability but can slow delivery speed."
     },
     "server": {
       "host": "127.0.0.1",
       "kind": "llama_cpp_server",
       "managed": true,
       "port": 8011
     },
     "trace": {
       "request_id": "example-clients-llama-cpp-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-llama-cpp-call-001.jsonl"
     }
   }

References
----------

- `llama.cpp llama-server docs <https://github.com/ggml-org/llama.cpp#llama-server>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
