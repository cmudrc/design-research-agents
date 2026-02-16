Workflows
=========

Workflow utilities provide reusable orchestration structures around tools and
agents.

Quick chooser
-------------

- Need reusable constructor-first step orchestration: start with ``Workflow``.
- Need low-level runtime execution control: use ``WorkflowRuntime`` directly.
- Need iterative orchestration loops: use ``LoopStep`` within ``WorkflowRuntime.run(...)``.
- Need planner + executor: use ``PlannerExecutorPattern``.
- Need proposal and critique loop: use ``ReflexionPattern``.
- Need intent-based delegate routing: use ``RouterPattern``.
- Need user-defined step graph: use ``Workflow`` with ``input_mode='schema'`` or
  ``input_mode='prompt'``.

Pages
-----

- :doc:`runtime_and_steps`
- :doc:`patterns`
- :doc:`composition_guide`

.. toctree::
   :maxdepth: 2
   :hidden:

   runtime_and_steps
   patterns
   composition_guide
