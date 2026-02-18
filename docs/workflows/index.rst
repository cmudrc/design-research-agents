Workflows
=========

Workflow utilities provide reusable orchestration structures around tools and
agents.

Quick chooser
-------------

- Need reusable constructor-first step orchestration: start with ``Workflow``.
- Need iterative orchestration loops: use ``LoopStep`` within ``Workflow``.
- Need low-level runtime execution control: use the internal
  ``design_research_agents.workflow.internal.workflow_runtime.WorkflowRuntime``
  engine only for advanced/internal extension work.
- Need planner + executor: use ``PlannerExecutorPattern``.
- Need proposal and critique loop: use ``ReflexionPattern``.
- Need intent-based delegate routing: use ``RouterPattern``.
- Need user-defined step graph: use ``Workflow`` with ``input_mode='schema'`` or
  ``input_mode='prompt'``.

Pages
-----

- :doc:`runtime_and_steps`
- :doc:`patterns`
- :doc:`networked_blackboard`
- :doc:`memory_and_rag`
- :doc:`reasoning_patterns`
- :doc:`composition_guide`
- :doc:`architecture_boundaries`

.. toctree::
   :maxdepth: 2
   :hidden:

   runtime_and_steps
   patterns
   networked_blackboard
   memory_and_rag
   reasoning_patterns
   composition_guide
   architecture_boundaries
