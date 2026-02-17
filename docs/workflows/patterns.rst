Workflow Patterns
=================

Available reusable workflow patterns:

- ``PlanExecuteWorkflow``
  - Planner + executor decomposition for complex tasks
- ``ProposeAndCritiqueWorkflow``
  - Iterative proposal/revision loops
- ``AgentRoutingWorkflow``
  - Intent-based delegation to selected agents
- ``DebatePattern``
  - Multi-round affirmative/negative debate with a judging pass
- ``PureToolWorkflow``
  - Reusable pure tool/logic graph with ``run(inputs=...)``
- ``MixedAgentWorkflow``
  - Reusable mixed graph with ``run(prompt=...)``

All workflow customization is constructor-first. Helper factory functions were
removed from ``design_research_agents.workflow.implementations``.

Workflows also support constructor-level run defaults (for request-id prefix,
base dependencies, and execution/failure policies where applicable).

Examples
--------

- ``examples/workflow/plan_execute.py``
- ``examples/workflow/propose_critic.py``
- ``examples/workflow/agent_routing.py``
- ``examples/workflow/debate_pattern.py``
- ``examples/workflow/pure_tool_workflow.py``
- ``examples/workflow/mixed_agent_workflow.py``
- ``examples/workflow/README.md``
