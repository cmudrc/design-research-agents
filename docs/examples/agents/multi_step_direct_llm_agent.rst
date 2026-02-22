Multi Step Direct LLM Agent
===========================

Source: ``examples/agents/multi_step_direct_llm_agent.py``

Introduction
------------

ReAct and Plan-and-Solve both motivate explicit multi-step reasoning loops instead of single-shot prompting,
and Toward Engineering AGI highlights why that structure matters for measurable engineering outcomes. This
example demonstrates a direct multi-step agent loop with traced iterations so design reasoning can be
inspected rather than inferred.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["MultiStepAgent.run(...)"]
       C --> D["WorkflowRuntime loop enforces continuation and max-step policy"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/agents/multi_step_direct_llm_agent.py
   :language: python
   :lines: 55-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/multi_step_direct_llm_agent.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "agents/multi_step_direct_llm_agent.py",
     "final_output": "42",
     "steps_executed": 2,
     "success": true,
     "terminated_reason": "stop:model",
     "trace": {
       "request_id": "example-multi-step-direct-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162205Z_example-multi-step-direct-design-001.jsonl"
     }
   }

References
----------

- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs <https://arxiv.org/abs/2509.16204>`_
