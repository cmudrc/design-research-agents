Reasoning Patterns
==================

Reasoning capabilities are exposed as reusable pattern implementations rather
than prompt-only conventions.

Available patterns
------------------

- ``ReflexionPattern``
  - Iterative propose/critique refinement.
  - Background references: `Reflexion <https://arxiv.org/abs/2303.11366>`_; `Self-Refine <https://arxiv.org/abs/2303.17651>`_. Conceptual grounding only; behavior is defined by repository contracts and implementation.
- ``TreeSearchPattern``
  - Generator + evaluator delegate orchestration with ``max_depth``, ``branch_factor``, and ``beam_width`` controls.
  - Background references: `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_. Conceptual grounding only; this implementation uses framework-native step orchestration.
- ``RAGPattern``
  - Retrieval-augmented reasoning with memory read/write workflow primitives.
  - Background references: `Retrieval-Augmented Generation (RAG) <https://arxiv.org/abs/2005.11401>`_. Conceptual grounding only; this pattern composes retrieval and context injection at workflow level.

Tree search output
------------------

``TreeSearchPattern`` returns:

.. code-block:: python

   {
       "best_candidate": {...},
       "best_score": float,
       "explored_nodes": int,
       "frontier_trace": [...],
   }

Examples
--------

- ``examples/patterns/tree_search.py``
- ``examples/patterns/propose_critic.py``
- ``examples/patterns/rag.py``
