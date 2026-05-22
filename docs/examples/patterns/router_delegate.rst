Router Delegate
===============

Source: ``examples/patterns/router_delegate.py``

Introduction
------------

RouteLLM motivates specialized route selection, AutoGen demonstrates multi-agent delegation patterns, and
Human-AI collaboration by design frames why explicit routing supports accountable coordination. This example
shows intent-based routing across direct and multi-step agents using a shared runtime surface.

.. note::

   This example's checked-in local ``LlamaCppServerLLMClient`` config uses a
   ``Qwen3-4B`` GGUF model. On lower-RAM machines, swap in a smaller local
   model or start with :doc:`../clients/ollama_local_client`.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``RouterDelegatePattern.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["RouterDelegatePattern.run(...)"]
       C --> D["router delegates to specialized agent surfaces"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/router_delegate.py
   :language: python
   :lines: 57-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/router_delegate.py

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

- `RouteLLM <https://arxiv.org/abs/2406.18665>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
