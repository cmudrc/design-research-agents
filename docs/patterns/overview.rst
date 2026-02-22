Pattern Overview
================

Available reusable pattern implementations:

- ``ConversationPattern``: two-speaker LLM conversation loops with per-speaker prompts and clients.
- ``DebatePattern``: multi-round affirmative/negative debate with a judging pass.
- ``PlannerExecutorPattern``: planner + executor decomposition for complex tasks.
- ``ReflexionPattern``: iterative proposal/revision loops.
- ``RouterPattern``: intent-based delegation to selected agents.
- ``NetworkedPattern``: round-based peer coordination with no central orchestrator.
- ``BlackboardPattern``: networked coordination with explicit shared board semantics.
- ``TreeSearchPattern``: generator/evaluator reasoning over beam-searched candidates.
- ``RagReasoningPattern``: retrieval-augmented reasoning via memory workflow steps.

Patterns are concrete workflow implementations, not construction primitives.
Use ``design_research_agents.workflow`` when composing new workflow graphs.

All pattern customization is constructor-first. Helper factory functions were
removed in favor of explicit class initialization.

Examples
--------

- ``examples/patterns/plan_execute.py``
- ``examples/patterns/propose_critic.py``
- ``examples/patterns/agent_routing.py``
- ``examples/patterns/debate_pattern.py``
- ``examples/patterns/conversation_pattern.py``
- ``examples/patterns/networked_blackboard.py``
- ``examples/patterns/tree_search.py``
- ``examples/patterns/rag_reasoning.py``
