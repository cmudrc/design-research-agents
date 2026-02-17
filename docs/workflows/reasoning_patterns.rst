Reasoning Patterns
==================

Reasoning capabilities are exposed as reusable workflow patterns rather than
prompt-only conventions.

Available patterns
------------------

- ``ReflexionPattern``
  - Iterative propose/critique refinement.
- ``TreeSearchPattern``
  - Generator + evaluator delegate orchestration with ``max_depth``, ``branch_factor``, and ``beam_width`` controls.
- ``RagReasoningPattern``
  - Retrieval-augmented reasoning with memory read/write workflow primitives.

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
