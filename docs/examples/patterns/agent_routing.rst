Agent Routing
=============

Source: ``examples/patterns/agent_routing.py``

Introduction
------------

RouteLLM motivates specialized route selection, AutoGen demonstrates multi-agent delegation patterns, and
Human-AI collaboration by design frames why explicit routing supports accountable coordination. This example
shows intent-based routing across direct and multi-step agents using a shared runtime surface.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``RouterPattern.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["RouterPattern.run(...)"]
       C --> D["router delegates to specialized agent surfaces"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/agent_routing.py
   :language: python
   :lines: 59-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/agent_routing.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "patterns/agent_routing.py",
     "final_output": {
       "char_count": 30,
       "line_count": 1,
       "unique_word_count": 4,
       "word_count": 4
     },
     "selected_alternative": null,
     "success": true,
     "terminated_reason": "max_steps_reached",
     "trace": {
       "request_id": "example-workflow-agent-routing-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162207Z_example-workflow-agent-routing-design-001.jsonl"
     }
   }

References
----------

- `RouteLLM <https://arxiv.org/abs/2406.18665>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
