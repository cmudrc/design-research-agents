Patterns
========

``design_research_agents.patterns`` contains prebuilt workflow
implementations. Each pattern is an implemented orchestration strategy built
from public workflow primitives.

Quick chooser
-------------

- Need planner + executor decomposition: ``PlanExecutePattern``.
- Need fixed two-role proposal/critique with approval gating: ``ProposeCriticPattern``.
- Need intent-based delegate routing: ``RouterDelegatePattern``.
- Need multi-round adversarial synthesis: ``DebatePattern``.
- Need two-speaker conversational iteration: ``TwoSpeakerConversationPattern``.
- Need decentralized peer rounds: ``RoundBasedCoordinationPattern``.
- Need shared-state convergence rounds: ``BlackboardPattern``.
- Need branch-and-score candidate search (beam or MCTS): ``TreeSearchPattern``.
- Need local numeric optimization over constrained state changes: ``SimulatedAnnealingPattern``.
<<<<<<< HEAD
- Need policy learning from reward signals in sequential design loops: :doc:`reinforcement_learning_starter`.
=======
- Need policy learning from reward signals in sequential design loops: :doc:`reinforcement_learning`.
>>>>>>> 77df08ad501aebf3994ba244d33bfff09fcd7477
- Need ordered multi-role refinement with evaluator score threshold: ``RalphLoopPattern``.
- Need independent fan-out drafts with evaluator-based best-of-N selection: ``NominalTeamPattern``.
- Need retrieval-augmented reasoning with write-back: ``RAGPattern``.
- Need to author a new implementation from scratch: :doc:`/workflows/index`.

Pages
-----

- :doc:`/examples/patterns/index`
- :doc:`overview`
- :doc:`reasoning`
- :doc:`coordination_patterns`
- :doc:`memory_and_rag`
<<<<<<< HEAD
- :doc:`reinforcement_learning_starter`
=======
- :doc:`reinforcement_learning`
>>>>>>> 77df08ad501aebf3994ba244d33bfff09fcd7477

.. toctree::
   :maxdepth: 2
   :hidden:

   overview
   reasoning
   coordination_patterns
   memory_and_rag
<<<<<<< HEAD
   reinforcement_learning_starter
=======
   reinforcement_learning
>>>>>>> 77df08ad501aebf3994ba244d33bfff09fcd7477
