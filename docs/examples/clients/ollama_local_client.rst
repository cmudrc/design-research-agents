Ollama Local Client
===================

Source: ``examples/clients/ollama_local_client.py``

Introduction
------------

Ollama operationalizes local model serving, the OpenAI Responses API provides a common contract surface, and
HELM underlines why comparable execution conditions matter in benchmarking. This example verifies the Ollama
client integration path under the project tracing/runtime conventions.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``OllamaLLMClient.generate(...)`` with a fixed
   ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["OllamaLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/clients/ollama_local_client.py
   :language: python
   :lines: 85-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/clients/ollama_local_client.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "base_url": "http://127.0.0.1:11434",
       "default_model": "qwen2.5:1.5b-instruct",
       "host": "127.0.0.1",
       "kind": "ollama",
       "max_retries": 2,
       "model_patterns": [
         "qwen2.5:*",
         "llama3:*"
       ],
       "name": "ollama-local-dev",
       "port": 11434
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "OllamaLLMClient",
     "default_model": "qwen2.5:1.5b-instruct",
     "example": "clients/ollama_local_client.py",
     "llm_call": {
       "prompt": "Give one sentence on when to use local model pull automation.",
       "response_has_text": true,
       "response_model": "qwen2.5:1.5b-instruct",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Use automated local pulls when startup reliability matters more than cold-start time."
     },
     "server": {
       "host": "127.0.0.1",
       "kind": "ollama",
       "managed": true,
       "port": 11434
     },
     "trace": {
       "request_id": "example-clients-ollama-local-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-ollama-local-call-001.jsonl"
     }
   }

References
----------

- `Ollama API Docs <https://docs.ollama.com/api>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
