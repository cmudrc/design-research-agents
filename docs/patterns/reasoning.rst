Reasoning Patterns
==================

Reasoning capabilities are exposed as reusable pattern implementations rather
than prompt-only conventions.

Available patterns
------------------

- ``ProposeCriticPattern``
  - Iterative two-role propose/critic refinement.
  - Requires an LLM client; ``tool_runtime`` is optional when delegates do not invoke tools.
  - Stop signal: critic ``approved`` boolean.
  - Returns ``ProposeCriticResult``, an ``ExecutionResult`` with direct ``proposal``, ``approved``, ``iterations``, ``reasoning``, and ``critique_iterations`` accessors.
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
- ``NominalTeamPattern``
  - Independent member fan-out followed by evaluator-driven best-of-N selection.
  - Stop signal: evaluator selects one candidate, or the run fails if no candidate can be selected.
  - Typical output focus: selected candidate + per-member generation diagnostics.
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
- Use ``NominalTeamPattern`` when you want diverse independent drafts first and only compare/select after generation.

Propose/critic result
---------------------

.. code-block:: python

   pattern = ProposeCriticPattern(llm_client=client, max_iterations=3)
   result = pattern.run("Draft a design rationale.")

   print(result.proposal)
   print(result.approved, result.iterations)
   print(result.reasoning)
   for critique in result.critique_iterations:
       print(critique["feedback"], critique["reasoning"])

``reasoning`` is the critic's optional, model-stated verdict rationale. The
default critic prompt requests a brief rationale, but custom critics may omit
it; in that case, ``result.reasoning`` and the iteration record's ``reasoning``
field are empty strings. This field supports auditing the relationship between
the verdict and the actionable ``feedback`` sent to the proposer. It is not
access to a model's hidden chain-of-thought and should not be treated as a
privileged record of its internal computation.

The common ``ExecutionResult`` fields and helpers remain available, including
``success``, ``final_output``, ``terminated_reason``, and ``summary()``.

Examples
--------

- ``examples/patterns/tree_search.py``
- ``examples/patterns/ralph_loop.py``
- ``examples/patterns/nominal_team.py``
- ``examples/patterns/propose_critic.py``
- ``examples/patterns/rag.py``
