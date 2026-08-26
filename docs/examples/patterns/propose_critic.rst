Propose Critic
==============

Source: ``examples/patterns/propose_critic.py``

Introduction
------------

Self-Refine and related critique/revise work motivate iterative self-critique loops, and Human-AI collaboration by design
explains why critique transparency is critical for trustworthy engineering decisions. This example
demonstrates a propose-critic refinement cycle with bounded iterations and structured run output.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``ProposeCriticPattern.run(...)`` with a fixed ``request_id``.
3. Read proposal, approval, iteration, and reasoning fields directly from the typed ``ProposeCriticResult``.
   ``reasoning`` is the critic's optional, model-stated verdict rationale, distinct from ``feedback``
   (the guidance sent back to the proposer). It is an observable explanation, not access to hidden
   chain-of-thought.
4. Print a compact JSON payload including trace metadata for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["ProposeCriticPattern.run(...)"]
       C --> D["proposal and critique turns iterate until stop criteria"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/propose_critic.py
   :language: python
   :lines: 63-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/patterns/propose_critic.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "approved": true,
     "error": null,
     "final_output": {
       "approved": true,
       "iterations": 1,
       "proposal": "<final proposal>",
       "reasoning": "<brief verdict rationale>"
     },
     "iterations": 1,
     "proposal": "<final proposal>",
     "reasoning": "<brief verdict rationale>",
     "success": true,
     "terminated_reason": "approved",
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }

References
----------

- `Self-Refine <https://arxiv.org/abs/2303.17651>`_
- `Reflexion <https://arxiv.org/abs/2303.11366>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
