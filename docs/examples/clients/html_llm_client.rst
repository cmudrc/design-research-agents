Html LLM Client
===============

Source: ``examples/clients/html_llm_client.py``

Introduction
------------

This example demonstrates the zero-dependency HTML stand-in client that ships with the framework. It is
useful for quickstarts, offline teaching, and trace smoke checks because it exercises the normal client
contract without depending on provider SDKs, local model runtimes, or network access.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``HTMLLLMClient.generate(...)`` with a fixed
   ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic inspection.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["HTMLLLMClient.generate(...)"]
       C --> D["LLMRequest/LLMResponse contracts stay unchanged"]
       C --> E["Tracer JSONL + console events"]
       D --> F["JSON payload"]
       E --> F

.. literalinclude:: ../../../examples/clients/html_llm_client.py
   :language: python
   :lines: 75-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/clients/html_llm_client.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "backend": {
       "base_url": null,
       "default_model": "html-standin-v1",
       "kind": "html",
       "max_retries": 0,
       "model_patterns": [
         "html-standin-v1"
       ],
       "name": "html-model"
     },
     "capabilities": {
       "json_mode": "none",
       "max_context_tokens": null,
       "streaming": true,
       "tool_calling": "none",
       "vision": false
     },
     "client_class": "HTMLLLMClient",
     "default_model": "html-standin-v1",
     "example": "clients/html_llm_client.py",
     "llm_call": {
       "prompt": "Wrap this design note in the stand-in HTML response.",
       "response_has_text": true,
       "response_model": "html-standin-v1",
       "response_provider": "html-model"
     },
     "server": null,
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }

References
----------

- `WHATWG HTML Living Standard <https://html.spec.whatwg.org/>`_
- `Python dataclasses <https://docs.python.org/3/library/dataclasses.html>`_
- `Python Protocols and structural subtyping <https://typing.python.org/en/latest/reference/protocols.html>`_
