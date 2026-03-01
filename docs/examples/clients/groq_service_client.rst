Groq Service Client
===================

Source: ``examples/clients/groq_service_client.py``

Introduction
------------

Groq hosted inference can provide low-latency responses for agent loops that still need standard chat-completion
semantics such as streaming and tool-call metadata. This example runs the Groq service client through the same
framework contracts used by other providers, with trace artifacts suitable for regression checks.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output for repeatable diagnostics.
2. Build one request using public APIs and execute ``GroqServiceLLMClient.generate(...)``.
3. Serialize the key response contract fields and backend metadata into one JSON payload.
4. Emit the payload with fixed request id metadata for deterministic documentation tests.

.. mermaid::

   flowchart LR
       A["Prompt input"] --> B["main(): tracing setup"]
       B --> C["GroqServiceLLMClient.generate(...)"]
       C --> D["LLMRequest and LLMResponse contracts"]
       C --> E["Tracer lifecycle events"]
       D --> F["Output payload"]
       E --> F
       F --> G["Printed JSON result"]

.. literalinclude:: ../../../examples/clients/groq_service_client.py
   :language: python
   :lines: 78-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/clients/groq_service_client.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "api_key_env": "GROQ_API_KEY",
       "base_url": "https://api.groq.com",
       "default_model": "llama-3.1-8b-instant",
       "kind": "groq_service",
       "max_retries": 3,
       "model_patterns": [
         "llama-3.1-8b-instant",
         "llama-3.1-*"
       ],
       "name": "groq-prod"
     },
     "capabilities": {
       "json_mode": "native",
       "max_context_tokens": null,
       "streaming": true,
       "tool_calling": "native",
       "vision": false
     },
     "client_class": "GroqServiceLLMClient",
     "default_model": "llama-3.1-8b-instant",
     "example": "clients/groq_service_client.py",
     "llm_call": {
       "prompt": "Provide one sentence on when teams should trade latency for review depth.",
       "response_has_text": true,
       "response_model": "llama-3.1-8b-instant",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Prefer deeper review when architectural choices are expensive to reverse."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-groq-service-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-groq-service-call-001.jsonl"
     }
   }

References
----------

- `Groq Python SDK repository <https://github.com/groq/groq-python>`_
- `Groq model catalog docs <https://console.groq.com/docs/models>`_
- `Groq chat completion docs <https://console.groq.com/docs/text-chat>`_
