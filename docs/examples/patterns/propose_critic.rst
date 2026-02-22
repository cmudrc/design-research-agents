Propose Critic
==============

Source: ``examples/patterns/propose_critic.py``

Introduction
------------

Reflexion and Self-Refine motivate iterative self-critique loops, and Human-AI collaboration by design
explains why critique transparency is critical for trustworthy engineering decisions. This example
demonstrates a propose-critic refinement cycle with bounded iterations and structured run output.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``ReflexionPattern.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["ReflexionPattern.run(...)"]
       C --> D["proposal and critique turns iterate until stop criteria"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/propose_critic.py
   :language: python
   :lines: 73-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/propose_critic.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "approved": true,
     "critique_iterations": [
       {
         "approved": false,
         "feedback": "Add more detail.",
         "iteration": 1,
         "proposal": "Draft v1: simple proposal.",
         "revision_goals": [
           "expand rationale"
         ]
       },
       {
         "approved": true,
         "feedback": "Looks good.",
         "iteration": 2,
         "proposal": "Draft v2: proposal with more detail.",
         "revision_goals": []
       }
     ],
     "error": null,
     "example": "patterns/propose_critic.py",
     "final_output": null,
     "proposal": "Draft v2: proposal with more detail.",
     "success": true,
     "terminated_reason": "approved",
     "trace": {
       "request_id": "example-workflow-propose-critic-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-workflow-propose-critic-design-001.jsonl"
     }
   }

References
----------

- `Reflexion <https://arxiv.org/abs/2303.11366>`_
- `Self-Refine <https://arxiv.org/abs/2303.17651>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
