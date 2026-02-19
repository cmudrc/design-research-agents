Workflows
=========

Workflow utilities provide reusable orchestration structures around tools and
agents. Pattern classes in this section are reusable agent-like orchestrators
built from the same public workflow primitives.

Quick chooser
-------------

- Need reusable constructor-first step orchestration: start with ``Workflow``.
- Need iterative orchestration loops: use ``LoopStep`` within ``Workflow``.
- Need low-level runtime execution control: compose a dedicated ``Workflow`` and
  use ``execution_mode`` / ``failure_policy`` run controls.
- Need planner + executor: use ``PlannerExecutorPattern``.
- Need proposal and critique loop: use ``ReflexionPattern``.
- Need two LLMs to converse with role-specific prompts/clients: use ``ConversationPattern``.
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
