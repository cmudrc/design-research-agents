Pattern Overview
================

Available reusable pattern implementations:

- ``TwoSpeakerConversationPattern``: two-speaker LLM conversation loops with per-speaker prompts and clients.
- ``DebatePattern``: multi-round affirmative/negative debate with a judging pass.
- ``PlanExecutePattern``: planner + executor decomposition for complex tasks.
- ``ProposeCriticPattern``: iterative proposal/revision loops.
- ``RouterDelegatePattern``: intent-based delegation to selected agents.
- ``RoundBasedCoordinationPattern``: round-based peer coordination with no central orchestrator.
- ``BlackboardPattern``: networked coordination with explicit shared board semantics.
- ``BeamSearchPattern``: generator/evaluator reasoning over beam-searched candidates.
- ``RAGPattern``: retrieval-augmented reasoning via memory workflow steps.

Patterns are concrete workflow implementations, not construction primitives.
Use ``design_research_agents.workflow`` when composing new workflow graphs.

All pattern customization is constructor-first. Helper factory functions were
removed in favor of explicit class initialization.

Examples
--------

- ``examples/patterns/plan_execute.py``
- ``examples/patterns/propose_critic.py``
- ``examples/patterns/router_delegate.py``
- ``examples/patterns/debate_pattern.py``
- ``examples/patterns/two_speaker_conversation.py``
- ``examples/patterns/coordination_patterns.py``
- ``examples/patterns/beam_search.py``
- ``examples/patterns/rag.py``
