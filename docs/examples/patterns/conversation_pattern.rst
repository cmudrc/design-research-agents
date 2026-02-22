Conversation Pattern
====================

Source: ``examples/patterns/conversation_pattern.py``

Introduction
------------

AutoGen-style multi-agent conversations can externalize reasoning roles, Human-AI collaboration by design
explains why role separation matters for oversight, and AI-assisted design synthesis work motivates
structured dialogue in design ideation. This example implements a two-agent conversation loop with trace
visibility at each turn.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``ConversationPattern.run(...)`` with a fixed
   ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["ConversationPattern.run(...)"]
       C --> D["turn-based conversation state drives each step"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/conversation_pattern.py
   :language: python
   :lines: 75-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/conversation_pattern.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "patterns/conversation_pattern.py",
     "final_output": {
       "message": "Prioritize the roller prototype first because it is simpler to fabricate; validate kernel breaka...
       "speaker": "Validation Engineer"
     },
     "participants": {
       "speaker_a": "Concept Designer",
       "speaker_b": "Validation Engineer"
     },
     "success": true,
     "terminated_reason": "completed",
     "trace": {
       "request_id": "example-workflow-conversation-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162207Z_example-workflow-conversation-design-001.jsonl"
     },
     "transcript_preview": [
       {
         "message": "Prototype a second concept with a peg-drum against a perforated concave, driven by gears to re...
         "speaker": "Concept Designer",
         "turn": 2
       },
       {
         "message": "Prioritize the roller prototype first because it is simpler to fabricate; validate kernel brea...
         "speaker": "Validation Engineer",
         "turn": 2
       }
     ],
     "turns_executed": 2
   }

References
----------

- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
- `AI-assisted design synthesis and human creativity in engineering education <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1714523/full>`_
