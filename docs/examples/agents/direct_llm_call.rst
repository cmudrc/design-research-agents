Direct LLM Call
===============

Source: ``examples/agents/direct_llm_call.py``

Introduction
------------

The default built-in path is the OpenAI-compatible HTTP client. This keeps the base install lightweight
while still talking to a real endpoint, whether that endpoint is local (for example llama.cpp, vLLM, or
SGLang) or remote behind an OpenAI-shaped gateway.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``DirectLLMCall.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["DirectLLMCall.run(...)"]
       C --> D["WorkflowRuntime executes one direct call"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/agents/direct_llm_call.py
   :language: python
   :lines: 51-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/direct_llm_call.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<example-specific payload>",
     "terminated_reason": "<string-or-null>",
     "error": null,
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }

References
----------

- `OpenAI API Reference <https://platform.openai.com/docs/api-reference/chat>`_
- `llama.cpp server documentation <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
