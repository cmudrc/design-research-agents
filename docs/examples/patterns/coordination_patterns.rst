Coordination Patterns
=====================

Source: ``examples/patterns/coordination_patterns.py``

Introduction
------------

Blackboard-system architecture motivates shared-state collaboration among specialized problem solvers,
AutoGen informs practical multi-agent implementation choices, and Human-AI collaboration by design clarifies
governance value in shared workspace reasoning. This example compares round-based coordination and
blackboard-specialized runs with explicit execution records.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``RoundBasedCoordinationPattern.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["RoundBasedCoordinationPattern.run(...)"]
       C --> D["blackboard workers contribute and aggregate shared state"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/coordination_patterns.py
   :language: python
   :lines: 65-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/coordination_patterns.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "round_based_coordination": {
       "success": true,
       "final_output": "<example-specific payload>",
       "terminated_reason": "<string-or-null>",
       "error": null,
       "trace": {
         "request_id": "<request-id>",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     },
     "blackboard": {
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
   }

References
----------

- `Blackboard System (Wikipedia) <https://en.wikipedia.org/wiki/Blackboard_system>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
