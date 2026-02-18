Architecture Boundaries
=======================

This page captures the intended separation of concerns for workflow orchestration.

Primary boundaries
------------------

- ``Workflow``: user-facing reusable facade for constructor-first step graphs with
  explicit input contracts and workflow-first output envelopes.
- ``WorkflowRuntime``: deterministic typed-step execution engine used by ``Workflow``.
- Patterns:
  - ``PlannerExecutorPattern``: planner output followed by iterative executor loop.
  - ``ReflexionPattern``: proposal/critique iterative refinement strategy.
  - ``RouterPattern``: selection + delegated execution strategy.

Step primitives
---------------

- ``LogicStep``: deterministic local logic and optional branching map.
- ``ToolStep``: single tool call via ``ToolRuntime``.
- ``AgentStep``: single delegate invocation via direct ``delegate`` object.
- ``LoopStep``: iterative nested workflow body with state transitions.

Allowed composition patterns
----------------------------

- Use ``Workflow`` to define reusable topology once and run repeatedly.
- Use ``LoopStep`` when iteration is a first-class part of orchestration.
- Keep prompt/model/tool policy concerns inside pattern classes or delegates, not in
  workflow scheduling internals.

Anti-patterns to avoid
----------------------

- Adding duplicate request-id/dependency helper functions in each pattern module.
- Encoding loop-state schema assumptions ad hoc inside multiple modules.
- Treating pattern classes as hidden internals; they are reusable strategy
  implementations built from workflow primitives.
- Coupling deterministic example harness behavior to script filenames only.
