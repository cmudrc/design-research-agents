Workflow Patterns
=================

Available reusable workflow patterns:

- ``PlannerExecutorPattern``
  - Planner + executor decomposition for complex tasks
- ``ReflexionPattern``
  - Iterative proposal/revision loops
- ``RouterPattern``
  - Intent-based delegation to selected agents
- ``Workflow``
  - Reusable user-defined graph with ``input_mode='schema'`` or
    ``input_mode='prompt'``

These workflow patterns are reference implementations built on first-class
workflow primitives: ``LogicStep``, ``ToolStep``, ``AgentStep``, and
``LoopStep``. You can reproduce and customize these
patterns directly with ``WorkflowRuntime`` rather than relying on hidden modes.

All workflow customization is constructor-first. Helper factory functions were
removed from ``design_research_agents.workflow.implementations``.

Workflows also support constructor-level run defaults (for request-id prefix,
base dependencies, and execution/failure policies where applicable).

Examples
--------

- ``examples/workflow/plan_execute.py``
- ``examples/workflow/propose_critic.py``
- ``examples/workflow/agent_routing.py``
- ``examples/workflow/pure_tool_workflow.py``
- ``examples/workflow/mixed_agent_workflow.py``
- ``examples/workflow/README.md``
