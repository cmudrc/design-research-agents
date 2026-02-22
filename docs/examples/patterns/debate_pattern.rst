Debate Pattern
==============

Source: ``examples/patterns/debate_pattern.py``

Introduction
------------

Multiagent Debate shows how adversarial dialogue can improve answer quality, AutoGen provides practical
orchestration motifs, and Human-AI collaboration by design situates debate outputs within reviewable
decision pipelines. This example runs a proposer-vs-critic debate pattern over shared tool/runtime
interfaces.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``DebatePattern.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["DebatePattern.run(...)"]
       C --> D["position agents debate before synthesis"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/debate_pattern.py
   :language: python
   :lines: 69-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/debate_pattern.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "patterns/debate_pattern.py",
     "final_output": {
       "synthesis": "Use local models for sensitive data and hosted APIs for burst capacity."
     },
     "rounds": [
       {
         "affirmative_argument": "Local models improve data control and predictable costs for many research workloads.",
         "negative_argument": "Hosted APIs can ship faster and often provide higher quality with less ops burden.",
         "round": 1
       }
     ],
     "success": true,
     "terminated_reason": "completed",
     "trace": {
       "request_id": "example-workflow-debate-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162207Z_example-workflow-debate-design-001.jsonl"
     },
     "verdict": {
       "rationale": "Both positions are compelling with different tradeoffs.",
       "synthesis": "Use local models for sensitive data and hosted APIs for burst capacity.",
       "winner": "tie"
     },
     "winner": "tie"
   }

References
----------

- `Multiagent Debate <https://arxiv.org/abs/2305.14325>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
