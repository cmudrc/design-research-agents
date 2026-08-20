OpenAI Service Client
=====================

Source: ``examples/clients/openai_service_client.py``

Introduction
------------

For hosted deployments, OpenAI platform docs and the Responses API capture production invocation behavior,
while function-calling guidance clarifies structured tool invocation expectations. This example shows the
direct OpenAI service client contract with traceable request/response handling.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``OpenAIServiceLLMClient.generate(...)`` with a
   fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["OpenAIServiceLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/clients/openai_service_client.py
   :language: python
   :lines: 83-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src DRA_EXAMPLE_LLM_MODE=deterministic python examples/clients/openai_service_client.py

This checkout-only command reproduces the documented output without a
live backend. For real installs, credentials, and backend-specific setup,
see :doc:`../../llm_clients/index`.

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "api_key_env": "OPENAI_API_KEY",
       "base_url": "https://api.openai.com/v1",
       "default_model": "gpt-4o-mini",
       "kind": "openai_service",
       "max_retries": 4,
       "model_patterns": [
         "gpt-4o-mini",
         "gpt-4o-*"
       ],
       "name": "openai-prod"
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "OpenAIServiceLLMClient",
     "default_model": "gpt-4o-mini",
     "example": "clients/openai_service_client.py",
     "llm_call": {
       "prompt": "In one sentence, when should engineering teams use multi-agent design critique?",
       "response_has_text": true,
       "response_model": "gpt-4o-mini",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Use multi-agent critique when decisions have high risk and need diverse failure analysis."
     },
     "llm_response_contract_preview": {
       "model": "gpt-4o-mini",
       "provider": "example-test-monkeypatch"
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-openai-service-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-openai-service-call-001.jsonl"
     }
   }

References
----------

- `OpenAI Platform Overview <https://platform.openai.com/docs/overview>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
