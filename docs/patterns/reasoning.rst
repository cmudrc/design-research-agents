Reasoning Patterns
==================

Reasoning capabilities are exposed as reusable pattern implementations rather
than prompt-only conventions.

Available patterns
------------------

- ``ProposeCriticPattern``
  - Iterative two-role propose/critic refinement.
  - Stop signal: critic ``approved`` boolean.
  - Typical output focus: latest approved proposal text + critique history.
  - Background references: `Self-Refine <https://arxiv.org/abs/2303.17651>`_; `Reflexion <https://arxiv.org/abs/2303.11366>`_. Conceptual grounding only; behavior is defined by repository contracts and implementation.
- ``TreeSearchPattern``
  - Generator + evaluator delegate orchestration with ``beam`` and ``mcts`` strategies.
  - Key controls: ``max_depth``, ``branch_factor``, ``beam_width``, ``search_strategy``, ``mcts_exploration_weight``, and ``simulation_budget``.
  - Stop signal: depth/search budget exhaustion or no expansions.
  - Typical output focus: best candidate + scored frontier diagnostics.
  - Background references: `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_. Conceptual grounding only; this implementation uses framework-native step orchestration.
- ``RalphLoopPattern``
  - Dynamic role-ordered loop with dedicated evaluator role, threshold-based stopping, and configurable selection strategy.
  - Stop signal: evaluator ``score`` crosses ``consensus_threshold`` (or max-iteration fallback).
  - Typical output focus: selected synthesis + per-role iteration history.
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

Pattern differentiation (quick)
-------------------------------

- Use ``ProposeCriticPattern`` when you want a strict two-role revision contract with explicit approval semantics.
- Use ``TreeSearchPattern`` when you want branching search over alternatives and deterministic score-driven pruning/selection.
- Use ``RalphLoopPattern`` when you want 3+ ordered roles and a separate evaluator scoring consensus quality each round.

Examples
--------

- ``examples/patterns/tree_search.py``
- ``examples/patterns/ralph_loop.py``
- ``examples/patterns/propose_critic.py``
- ``examples/patterns/rag.py``
