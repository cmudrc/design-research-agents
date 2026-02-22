Architecture Boundaries
=======================

This page captures the intended separation of concerns for workflow orchestration.

Primary boundaries
------------------

- ``design_research_agents.workflow``: public construction surface for composing
  new workflow implementations from typed step objects.
- ``Workflow``: reusable facade for constructor-first step graphs with explicit
  input contracts and workflow-first output envelopes.
- ``WorkflowRuntime``: deterministic typed-step execution engine used by
  ``Workflow``.
- ``design_research_agents.patterns``: public prebuilt implementations built
  from workflow primitives.

Step primitives
---------------

- ``LogicStep``: deterministic local logic and optional branching map.
- ``ToolStep``: single tool call via ``ToolRuntime``.
- ``AgentStep``: single delegate invocation via direct ``delegate`` object.
- ``LoopStep``: iterative nested workflow body with state transitions.
- ``DelegateBatchStep``: explicit multi-delegate fan-out step.
- ``MemoryReadStep`` and ``MemoryWriteStep``: workflow-native memory primitives.

Allowed composition patterns
----------------------------

- Use ``Workflow`` + step primitives to define reusable topology once and run
  repeatedly.
- Use ``design_research_agents.patterns`` when you want a prebuilt workflow
  implementation instead of authoring a graph from scratch.
- Keep prompt/model/tool policy concerns in step callbacks/delegates or pattern
  classes, not in workflow scheduling internals.

Anti-patterns to avoid
----------------------

- Mixing prebuilt pattern exports into the workflow construction module.
- Encoding loop-state schema assumptions ad hoc inside multiple modules.
- Adding duplicate request-id/dependency helper functions across prebuilt
  pattern implementations.
- Coupling deterministic example harness behavior to script filenames only.
