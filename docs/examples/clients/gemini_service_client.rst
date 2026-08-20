Gemini Service Client
=====================

Source: ``examples/clients/gemini_service_client.py``

Introduction
------------

Gemini hosted inference is useful when teams want multimodel experimentation through one provider SDK, while
keeping request payloads under the framework's provider-neutral LLM contracts. This example exercises the
Gemini service client path with trace capture and deterministic output support for CI.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console sinks so each run emits machine-readable traces.
2. Build runtime inputs through public package APIs and invoke ``GeminiServiceLLMClient.generate(...)``.
3. Construct ``LLMRequest`` payload fields and execute one representative remote-style call.
4. Print a compact JSON payload that includes trace metadata for docs and deterministic tests.

.. mermaid::

   flowchart LR
       A["Prompt input"] --> B["main(): tracing setup"]
       B --> C["GeminiServiceLLMClient.generate(...)"]
       C --> D["LLMRequest and LLMResponse contracts"]
       C --> E["Tracer JSONL + console events"]
       D --> F["Output payload"]
       E --> F
       F --> G["Printed JSON result"]

.. literalinclude:: ../../../examples/clients/gemini_service_client.py
   :language: python
   :lines: 77-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src DRA_EXAMPLE_LLM_MODE=deterministic python examples/clients/gemini_service_client.py

This checkout-only command reproduces the documented output without a
live backend. For real installs, credentials, and backend-specific setup,
see :doc:`../../llm_clients/index`.

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "api_key_env": "GOOGLE_API_KEY",
       "default_model": "gemini-2.5-flash",
       "kind": "gemini_service",
       "max_retries": 3,
       "model_patterns": [
         "gemini-2.5-flash",
         "gemini-2.5-*"
       ],
       "name": "gemini-prod"
     },
     "capabilities": {
       "json_mode": "native",
       "max_context_tokens": null,
       "streaming": true,
       "tool_calling": "none",
       "vision": false
     },
     "client_class": "GeminiServiceLLMClient",
     "default_model": "gemini-2.5-flash",
     "example": "clients/gemini_service_client.py",
     "llm_call": {
       "prompt": "In one sentence, when should engineers run an explicit design pre-mortem?",
       "response_has_text": true,
       "response_model": "gemini-2.5-flash",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Run a design pre-mortem before committing architecture changes with high uncertainty or safety risk."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-gemini-service-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-gemini-service-call-001.jsonl"
     }
   }

References
----------

- `Google Gen AI Python SDK docs <https://googleapis.github.io/python-genai/>`_
- `Gemini API key setup guide <https://ai.google.dev/gemini-api/docs/api-key>`_
- `Gemini API model docs <https://ai.google.dev/gemini-api/docs/models>`_
