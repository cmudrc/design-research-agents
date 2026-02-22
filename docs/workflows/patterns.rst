Workflow Patterns
=================

Available reusable workflow patterns:
- ``ConversationPattern``: two-speaker LLM conversation loops with per-speaker prompts and clients.
- ``DebatePattern``: multi-round affirmative/negative debate with a judging pass
- ``PlannerExecutorPattern``: planner + executor decomposition for complex tasks.
- ``ReflexionPattern``: iterative proposal/revision loops.
- ``RouterPattern``: intent-based delegation to selected agents.
- ``NetworkedPattern``: round-based peer coordination with no central orchestrator.
- ``BlackboardPattern``: networked coordination with explicit shared board semantics.
- ``TreeSearchPattern``: generator/evaluator reasoning over beam-searched candidates.
- ``RagReasoningPattern``: retrieval-augmented reasoning via memory workflow steps.
- ``Workflow``: reusable user-defined graph with inferred input mode from ``input_schema``.

These workflow patterns are reference implementations built on first-class
workflow primitives: ``LogicStep``, ``ToolStep``, ``AgentStep``, and
``LoopStep``. You can reproduce and customize these patterns with ``Workflow``
plus step primitives rather than relying on hidden modes. ``WorkflowRuntime``
remains an internal engine for advanced/internal extension work.

All workflow customization is constructor-first. Helper factory functions were
removed in favor of the public ``Workflow`` + step-object composition model.

Workflows also support constructor-level run defaults (for request-id prefix,
base dependencies, and execution/failure policies where applicable).

Examples
--------

- ``examples/patterns/plan_execute.py``
- ``examples/patterns/propose_critic.py``
- ``examples/patterns/agent_routing.py``
- ``examples/patterns/debate_pattern.py``
- ``examples/patterns/conversation_pattern.py``
- ``examples/workflow/workflow_schema_mode.py``
- ``examples/workflow/workflow_prompt_mode.py``
- ``examples/patterns/README.md``
