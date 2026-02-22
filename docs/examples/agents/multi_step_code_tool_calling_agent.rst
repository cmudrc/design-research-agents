Multi Step Code Tool Calling Agent
==================================

Source: ``examples/agents/multi_step_code_tool_calling_agent.py``

Introduction
------------

ReAct and Toolformer motivate external action for model reasoning, while AutoGen highlights how
multi-agent/tool ecosystems depend on explicit execution boundaries. This example focuses on code-tool
calling so you can study how executable outputs are requested, validated, and traced in a controlled loop.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
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

.. literalinclude:: ../../../examples/agents/multi_step_code_tool_calling_agent.py
   :language: python
   :lines: 61-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/multi_step_code_tool_calling_agent.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "agents/multi_step_code_tool_calling_agent.py",
     "final_output": {
       "char_count": 1154,
       "line_count": 7,
       "unique_word_count": 100,
       "word_count": 135
     },
     "step_outputs_count": 1,
     "steps_executed": 1,
     "success": true,
     "terminated_reason": "continuation_stopped:model",
     "tool_results_count": 1,
     "trace": {
       "request_id": "example-multi-step-code-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162205Z_example-multi-step-code-design-001.jsonl"
     }
   }

References
----------

- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
