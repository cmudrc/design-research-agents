Workflow Patterns
=================

Available reusable workflow patterns:

- ``PlanExecuteWorkflow``
  - Planner + executor decomposition for complex tasks
- ``ProposeAndCritiqueWorkflow``
  - Iterative proposal/revision loops
- ``AgentRoutingWorkflow``
  - Intent-based delegation to selected agents
- ``PureToolWorkflow``
  - Reusable pure tool/logic graph with ``run(inputs=...)``
- ``MixedAgentWorkflow``
  - Reusable mixed graph with ``run(prompt=...)``

Examples
--------

- ``examples/workflow/plan_execute.py``
- ``examples/workflow/propose_critic.py``
- ``examples/workflow/agent_routing.py``
- ``examples/workflow/pure_tool_workflow.py``
- ``examples/workflow/mixed_agent_workflow.py``
- ``examples/workflow/README.md``
