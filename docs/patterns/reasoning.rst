Reasoning Patterns
==================

Reasoning capabilities are exposed as reusable pattern implementations rather
than prompt-only conventions.

Available patterns
------------------

- ``ProposeCriticPattern``
  - Iterative propose/critic refinement.
  - Background references: `Self-Refine <https://arxiv.org/abs/2303.17651>`_; `Reflexion <https://arxiv.org/abs/2303.11366>`_. Conceptual grounding only; behavior is defined by repository contracts and implementation.
- ``TreeSearchPattern``
  - Generator + evaluator delegate orchestration with ``beam`` and ``mcts`` strategies.
  - Key controls: ``max_depth``, ``branch_factor``, ``beam_width``, ``search_strategy``, ``mcts_exploration_weight``, and ``simulation_budget``.
  - Background references: `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_. Conceptual grounding only; this implementation uses framework-native step orchestration.
- ``RalphLoopPattern``
  - Dynamic role-ordered loop with dedicated evaluator role, threshold-based stopping, and configurable selection strategy.
- ``RAGPattern``
  - Retrieval-augmented reasoning with memory read/write workflow primitives.
  - Background references: `Retrieval-Augmented Generation (RAG) <https://arxiv.org/abs/2005.11401>`_. Conceptual grounding only; this pattern composes retrieval and context injection at workflow level.

Tree search output
------------------

``TreeSearchPattern`` returns:

.. code-block:: python

   {
       "final_output": {
           "best_candidate": {...},
           "best_score": float,
       },
       "details": {
           "explored_nodes": int,
           "frontier_trace": [...],
       },
       "terminated_reason": str,
   }

Examples
--------

- ``examples/patterns/tree_search.py``
- ``examples/patterns/ralph_loop.py``
- ``examples/patterns/propose_critic.py``
- ``examples/patterns/rag.py``
