Workflow Examples
=================

These scripts demonstrate workflow primitives and reusable patterns for
engineering-design coordination.

Core ``Workflow`` and step primitives
-------------------------------------

- ``examples/workflow/workflow_runtime.py``
  Observe: successful logic-only run and trace artifact path.
  Public API: ``Workflow``, ``LogicStep``.
- ``examples/workflow/workflow_runtime_loop_step.py``
  Observe: loop termination state and final output summary.
  Public API: ``LoopStep``.
- ``examples/workflow/workflow_schema_mode.py``
  Observe: strict/relaxed run comparison and report artifact paths.
  Public API: ``Workflow``, ``ToolStep``.
- ``examples/workflow/workflow_prompt_mode.py``
  Observe: route-based branching (agent vs template path).
  Public API: ``Workflow``, ``AgentStep``.
- ``examples/workflow/workflow_model_step_design_tradeoff.py``
  Observe: ``ModelStep`` parsed output and finalized tradeoff text.
  Public API: ``ModelStep``.
- ``examples/workflow/workflow_delegate_and_memory_steps.py``
  Observe: memory seed/read and delegate batch outputs in one run.
  Public API: ``DelegateBatchStep``, ``MemoryReadStep``, ``MemoryWriteStep``.

Reusable patterns
-----------------

- ``examples/workflow/plan_execute.py``
  Public API: ``PlannerExecutorPattern``.
- ``examples/workflow/propose_critic.py``
  Public API: ``ReflexionPattern``.
- ``examples/workflow/agent_routing.py``
  Public API: ``RouterPattern``.
- ``examples/workflow/debate_pattern.py``
  Public API: ``DebatePattern``.
- ``examples/workflow/conversation_pattern.py``
  Public API: ``ConversationPattern``.
- ``examples/workflow/networked_blackboard.py``
  Public API: ``NetworkedPattern``, ``BlackboardPattern``.
- ``examples/workflow/tree_search.py``
  Public API: ``TreeSearchPattern``.
- ``examples/workflow/rag_reasoning.py``
  Public API: ``RagReasoningPattern``.

For each script, observe ``terminated_reason`` and ``trace.trace_path``.

Observed local run snippets (2026-02-21)
----------------------------------------

``examples/workflow/workflow_runtime.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/workflow/workflow_runtime.py

Observed stdout:

.. code-block:: python

   {
     "success": True,
     "execution_order": ["design_runtime_ready"],
     "final_output": {
       "message": "Design runtime orchestration validated.",
       "check": "workflow-runtime-ready"
     },
     "trace": {
       "trace_path": "artifacts/examples/traces/run_<timestamp>_example-workflow-runtime-design-001.jsonl"
     }
   }

``examples/workflow/workflow_delegate_and_memory_steps.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/workflow/workflow_delegate_and_memory_steps.py

Observed stdout:

.. code-block:: json

   {
     "example": "workflow/workflow_delegate_and_memory_steps.py",
     "execution_order": ["seed_constraints", "read_constraints", "peer_batch", "finalize"],
     "final_output": {"constraints_found": 2, "delegate_calls": 2},
     "success": true
   }

``examples/workflow/networked_blackboard.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/workflow/networked_blackboard.py

Observed stdout:

.. code-block:: json

   {
     "blackboard_pattern": {"rounds_executed": 3, "message_count": 6, "success": true},
     "networked_pattern": {"rounds_executed": 2, "message_count": 0, "success": true}
   }
