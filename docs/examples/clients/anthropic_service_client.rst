Anthropic Service Client
========================

Source: ``examples/clients/anthropic_service_client.py``

Introduction
------------

Anthropic hosted inference is useful when teams want strong instruction-following and tool-use support from one
managed API while keeping application code on provider-neutral LLM contracts. This example exercises the
Anthropic service client path with trace capture and deterministic output support for CI.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console sinks so each run emits machine-readable traces.
2. Build runtime inputs through public package APIs and invoke ``AnthropicServiceLLMClient.generate(...)``.
3. Construct ``LLMRequest`` payload fields and execute one representative remote-style call.
4. Print a compact JSON payload that includes trace metadata for docs and deterministic tests.

.. mermaid::

   flowchart LR
       A["Prompt input"] --> B["main(): tracing setup"]
       B --> C["AnthropicServiceLLMClient.generate(...)"]
       C --> D["LLMRequest and LLMResponse contracts"]
       C --> E["Tracer JSONL + console events"]
       D --> F["Output payload"]
       E --> F
       F --> G["Printed JSON result"]

.. literalinclude:: ../../../examples/clients/anthropic_service_client.py
   :language: python
   :lines: 78-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/clients/anthropic_service_client.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "api_key_env": "ANTHROPIC_API_KEY",
       "base_url": "https://api.anthropic.com",
       "default_model": "claude-3-5-haiku-latest",
       "kind": "anthropic_service",
       "max_retries": 3,
       "model_patterns": [
         "claude-3-5-haiku-latest",
         "claude-3-5-*"
       ],
       "name": "anthropic-prod"
     },
     "capabilities": {
       "json_mode": "native",
       "max_context_tokens": null,
       "streaming": true,
       "tool_calling": "native",
       "vision": false
     },
     "client_class": "AnthropicServiceLLMClient",
     "default_model": "claude-3-5-haiku-latest",
     "example": "clients/anthropic_service_client.py",
     "llm_call": {
       "prompt": "In one sentence, when should teams run architecture red-team reviews?",
       "response_has_text": true,
       "response_model": "claude-3-5-haiku-latest",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Run architecture red-team reviews before committing high-impact changes with uncertain failure modes."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-anthropic-service-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-anthropic-service-call-001.jsonl"
     }
   }

References
----------

- `Anthropic API docs <https://docs.anthropic.com/en/api/overview>`_
- `Anthropic Python SDK repository <https://github.com/anthropics/anthropic-sdk-python>`_
- `Anthropic tool use docs <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview>`_
