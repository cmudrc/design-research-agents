Multi Step JSON Tool Calling 1d Optimization
============================================

Source: ``examples/optimization/multi_step_json_tool_calling_1d_optimization.py``

Introduction
------------

Practical Bayesian optimization motivates iterative search over expensive objective evaluations, while
Toolformer and Plan-and-Solve motivate explicit action/reason loops for model-guided exploration. This
example operationalizes that idea as a JSON tool-calling optimization workflow with traceable proposals and
evaluations.

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
       C --> D["optimization loop combines callable tools with continuation control"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/optimization/multi_step_json_tool_calling_1d_optimization.py
   :language: python
   :lines: 161-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/optimization/multi_step_json_tool_calling_1d_optimization.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "agent": "MultiStepAgent(mode=json)",
     "best_seen": {
       "best_history_index": 3,
       "best_objective": 0.0,
       "best_x": 0.0
     },
     "example": "optimization/multi_step_json_tool_calling_1d_optimization.py",
     "final_output": {
       "best_objective": 0.0,
       "best_x": 0.0,
       "f_x": 0.0,
       "history": [
         3.0,
         2.0,
         1.0,
         0.0
       ],
       "improved": true,
       "previous_f_x": 1.0,
       "previous_x": 1.0,
       "x": 0.0
     },
     "history": [
       3.0,
       2.0,
       1.0,
       0.0
     ],
     "memory_tail": [
       {
         "kind": "action",
         "step": 2,
         "tool_input": {
           "step": 1
         },
         "tool_name": "optimizer.decrease_x"
       },
       {
         "error": "Step execution failed.",
         "final_output": {
           "best_objective": 1.0,
           "best_x": 1.0,
           "f_x": 1.0,
           "history": [
             3.0,
             2.0,
             1.0
           ],
           "improved": true,
           "previous_f_x": 4.0,
           "previous_x": 2.0,
           "x": 1.0
         },
         "kind": "observation",
         "step": 2,
         "success": true
       },
       {
         "continue": true,
         "kind": "thought",
         "source": "model",
         "step": 3,
         "text": "One more decrease should reach zero."
       },
       {
         "kind": "action",
         "step": 3,
         "tool_input": {
           "step": 1
         },
         "tool_name": "optimizer.decrease_x"
       },
       {
         "error": "Step execution failed.",
         "final_output": {
           "best_objective": 0.0,
           "best_x": 0.0,
           "f_x": 0.0,
           "history": [
             3.0,
             2.0,
             1.0,
             0.0
           ],
           "improved": true,
           "previous_f_x": 1.0,
           "previous_x": 1.0,
           "x": 0.0
         },
         "kind": "observation",
         "step": 3,
         "success": true
       },
       {
         "continue": false,
         "kind": "thought",
         "source": "model",
         "step": 4,
         "text": "No better one-step move remains."
       }
     ],
     "objective": "x^2",
     "objective_history": [
       9.0,
       4.0,
       1.0,
       0.0
     ],
     "steps_executed": 3,
     "success": true,
     "terminated_reason": "continuation_stopped:model",
     "tool_results_count": 3,
     "trace": {
       "request_id": "example-optimization-json-tool-calling-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }

References
----------

- `Practical Bayesian Optimization of Machine Learning Algorithms <https://arxiv.org/abs/1012.2599>`_
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
