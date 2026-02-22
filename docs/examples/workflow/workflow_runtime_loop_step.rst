Workflow Runtime Loop Step
==========================

Source: ``examples/workflow/workflow_runtime_loop_step.py``

Introduction
------------

Tree of Thoughts and ReAct each motivate iterative reasoning with explicit state updates, and AutoGen
provides a practical framing for orchestrating repeated loop actions. This example demonstrates loop-step
execution in the workflow runtime, including bounded iteration behavior and trace emission.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["Workflow.run(...)"]
       C --> D["WorkflowRuntime schedules step graph (LogicStep, LoopStep)"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/workflow/workflow_runtime_loop_step.py
   :language: python
   :lines: 340-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/workflow_runtime_loop_step.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "workflow/workflow_runtime_loop_step.py",
     "execution_order": [
       "design_counter_loop"
     ],
     "final_output": {
       "final_state": {
         "counter": 3
       },
       "iteration_results": [
         {
           "execution_order": [
             "increment",
             "snapshot"
           ],
           "metadata": {
             "artifact_count": 0,
             "dependency_keys": [],
             "execution_mode": "sequential",
             "failure_policy": "skip_dependents",
             "memory_enabled": false,
             "request_id": "example-workflow-loop-design-001:workflow:design_counter_loop:loop:1",
             "runtime": "workflow",
             "step_count": 2
           },
           "model_response": null,
           "output": {
             "artifacts": [],
             "final_output": {
               "counter": 1,
               "status": "looping"
             },
             "workflow": {
               "execution_order": [
                 "increment",
                 "snapshot"
               ],
               "step_results": {
                 "increment": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 1
                   },
                   "status": "completed",
                   "step_id": "increment",
                   "success": true
                 },
                 "snapshot": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 1,
                     "status": "looping"
                   },
                   "status": "completed",
                   "step_id": "snapshot",
                   "success": true
                 }
               },
               "success": true
             }
           },
           "step_results": {
             "increment": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 1
               },
               "status": "completed",
               "step_id": "increment",
               "success": true
             },
             "snapshot": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 1,
                 "status": "looping"
               },
               "status": "completed",
               "step_id": "snapshot",
               "success": true
             }
           },
           "success": true,
           "tool_results": []
         },
         {
           "execution_order": [
             "increment",
             "snapshot"
           ],
           "metadata": {
             "artifact_count": 0,
             "dependency_keys": [],
             "execution_mode": "sequential",
             "failure_policy": "skip_dependents",
             "memory_enabled": false,
             "request_id": "example-workflow-loop-design-001:workflow:design_counter_loop:loop:2",
             "runtime": "workflow",
             "step_count": 2
           },
           "model_response": null,
           "output": {
             "artifacts": [],
             "final_output": {
               "counter": 2,
               "status": "looping"
             },
             "workflow": {
               "execution_order": [
                 "increment",
                 "snapshot"
               ],
               "step_results": {
                 "increment": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 2
                   },
                   "status": "completed",
                   "step_id": "increment",
                   "success": true
                 },
                 "snapshot": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 2,
                     "status": "looping"
                   },
                   "status": "completed",
                   "step_id": "snapshot",
                   "success": true
                 }
               },
               "success": true
             }
           },
           "step_results": {
             "increment": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 2
               },
               "status": "completed",
               "step_id": "increment",
               "success": true
             },
             "snapshot": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 2,
                 "status": "looping"
               },
               "status": "completed",
               "step_id": "snapshot",
               "success": true
             }
           },
           "success": true,
           "tool_results": []
         },
         {
           "execution_order": [
             "increment",
             "snapshot"
           ],
           "metadata": {
             "artifact_count": 0,
             "dependency_keys": [],
             "execution_mode": "sequential",
             "failure_policy": "skip_dependents",
             "memory_enabled": false,
             "request_id": "example-workflow-loop-design-001:workflow:design_counter_loop:loop:3",
             "runtime": "workflow",
             "step_count": 2
           },
           "model_response": null,
           "output": {
             "artifacts": [],
             "final_output": {
               "counter": 3,
               "status": "threshold_met"
             },
             "workflow": {
               "execution_order": [
                 "increment",
                 "snapshot"
               ],
               "step_results": {
                 "increment": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 3
                   },
                   "status": "completed",
                   "step_id": "increment",
                   "success": true
                 },
                 "snapshot": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 3,
                     "status": "threshold_met"
                   },
                   "status": "completed",
                   "step_id": "snapshot",
                   "success": true
                 }
               },
               "success": true
             }
           },
           "step_results": {
             "increment": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 3
               },
               "status": "completed",
               "step_id": "increment",
               "success": true
             },
             "snapshot": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 3,
                 "status": "threshold_met"
               },
               "status": "completed",
               "step_id": "snapshot",
               "success": true
             }
           },
           "success": true,
           "tool_results": []
         }
       ],
       "iterations": 10,
       "iterations_executed": 3,
       "success": true,
       "terminated_reason": "condition_stopped"
     },
     "loop_status": "condition_stopped",
     "success": true,
     "terminated_reason": null,
     "trace": {
       "request_id": "example-workflow-loop-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-loop-design-001.jsonl"
     }
   }

References
----------

- `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
