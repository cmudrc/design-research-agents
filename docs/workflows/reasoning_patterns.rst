Reasoning Patterns
==================

Reasoning capabilities are exposed as reusable workflow patterns rather than
prompt-only conventions.

Available patterns
------------------

- ``ReflexionPattern``
  - Iterative propose/critique refinement.
  - Background references: `Reflexion <https://arxiv.org/abs/2303.11366>`_; `Self-Refine <https://arxiv.org/abs/2303.17651>`_. Conceptual grounding only; this pattern is defined by workflow contracts in this repository.
- ``TreeSearchPattern``
  - Generator + evaluator delegate orchestration with ``max_depth``, ``branch_factor``, and ``beam_width`` controls.
  - Background references: `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_. Conceptual grounding only; this implementation uses framework-native step orchestration.
- ``RagReasoningPattern``
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

- ``examples/workflow/tree_search.py``
- ``examples/workflow/propose_critic.py``
- ``examples/workflow/rag_reasoning.py``
