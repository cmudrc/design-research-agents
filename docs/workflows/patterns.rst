Workflow Patterns
=================

Available reusable workflow patterns:
- ``DebatePattern``: multi-round affirmative/negative debate with a judging pass
- ``PlannerExecutorPattern``: planner + executor decomposition for complex tasks.
- ``ReflexionPattern``: iterative proposal/revision loops.
- ``RouterPattern``: intent-based delegation to selected agents.
- ``Workflow``: reusable user-defined graph with ``input_mode='schema'`` or ``input_mode='prompt'``.

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
- ``examples/workflow/debate_pattern.py``
- ``examples/workflow/workflow_schema_mode.py``
- ``examples/workflow/workflow_prompt_mode.py``
- ``examples/workflow/README.md``
