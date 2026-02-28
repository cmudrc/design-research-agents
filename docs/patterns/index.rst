Patterns
========

``design_research_agents.patterns`` contains prebuilt workflow
implementations. Each pattern is an implemented orchestration strategy built
from public workflow primitives.

Quick chooser
-------------

- Need planner + executor decomposition: ``PlanExecutePattern``.
- Need iterative proposal/critique refinement: ``ProposeCriticPattern``.
- Need intent-based delegate routing: ``RouterDelegatePattern``.
- Need multi-round adversarial synthesis: ``DebatePattern``.
- Need two-speaker conversational iteration: ``TwoSpeakerConversationPattern``.
- Need decentralized peer rounds: ``RoundBasedCoordinationPattern``.
- Need shared-state convergence rounds: ``BlackboardPattern``.
- Need candidate expansion + scoring search: ``BeamSearchPattern``.
- Need retrieval-augmented reasoning with write-back: ``RAGPattern``.
- Need to author a new implementation from scratch: :doc:`/workflows/index`.

Pages
-----

- :doc:`/examples/patterns/index`
- :doc:`overview`
- :doc:`reasoning`
- :doc:`coordination_patterns`
- :doc:`memory_and_rag`

.. toctree::
   :maxdepth: 2
   :hidden:

   overview
   reasoning
   coordination_patterns
   memory_and_rag
