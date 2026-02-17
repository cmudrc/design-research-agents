Architecture Boundaries
=======================

This page captures the intended separation of concerns for workflow orchestration.

Primary boundaries
------------------

- ``AgentRuntime``: react-only execution runtime for agent model APIs.
- ``WorkflowRuntime``: deterministic typed-step orchestration executor.
- ``Workflow``: user-facing reusable facade for constructor-first step graphs with
  explicit input contracts.
- Patterns:
  - ``PlannerExecutorPattern``: planner output followed by iterative executor loop.
  - ``ReflexionPattern``: proposal/critique iterative refinement strategy.
  - ``RouterPattern``: selection + delegated execution strategy.

Step primitives
---------------

- ``LogicStep``: deterministic local logic and optional branching map.
- ``ToolStep``: single tool call via ``ToolRuntime``.
- ``AgentStep``: single delegate invocation through configured agents.
- ``LoopStep``: iterative nested workflow body with state transitions.

Allowed composition patterns
----------------------------

- Use ``Workflow`` to define reusable topology once and run repeatedly.
- Use ``LoopStep`` when iteration is a first-class part of orchestration.
- Keep prompt/model/tool policy concerns inside pattern classes or delegates, not in
  ``WorkflowRuntime`` scheduling internals.

Anti-patterns to avoid
----------------------

- Reintroducing orchestration modes directly into ``AgentRuntime``.
- Adding duplicate request-id/dependency helper functions in each pattern module.
- Encoding loop-state schema assumptions ad hoc inside multiple modules.
- Treating pattern classes as hidden internals; they are reusable strategy
  implementations built from workflow primitives.
- Coupling deterministic example harness behavior to script filenames only.
