Workflows
=========

Workflow utilities provide reusable orchestration structures around tools and
agents.

Quick chooser
-------------

- Need generic typed step orchestration: start with ``WorkflowRuntime``.
- Need planner + executor: use ``PlanExecuteWorkflow``.
- Need proposal and critique loop: use ``ProposeAndCritiqueWorkflow``.
- Need intent-based delegate routing: use ``AgentRoutingWorkflow``.
- Need user-defined tool/logic graph: use ``PureToolWorkflow``.
- Need mixed logic/agent/tool graph: use ``MixedAgentWorkflow``.

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
