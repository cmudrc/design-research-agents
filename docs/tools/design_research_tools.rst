Design Research Tools
=====================

This page summarizes the design- and engineering-oriented core tools that pair
well with structured JSON tool-calling agents.

Python sandbox
--------------

- ``python.sandbox`` executes guarded Python snippets with:
  - no imports
  - no filesystem/network/process access
  - timeout and stdout-size limits
  - JSON-serializable result enforcement

Memory tools (SQLite-first)
---------------------------

- ``memory.search``
- ``memory.write``
- ``memory.stats``

Store resolution order:

1. ``dependencies["memory_store"]`` when provided and protocol-compatible.
2. ``db_path`` from tool input when provided.
3. Default ``artifacts/memory/memory.sqlite3`` path.

``memory.search`` returns ``retrieval_mode`` as:

- ``lexical`` when vector scores are unavailable.
- ``hybrid_vector`` when vector similarity contributes to ranking.

Evaluation tools
----------------

- ``eval.decision_matrix`` for weighted multi-criteria ranking.
- ``eval.pairwise_rank`` for pairwise comparisons using Copeland scoring.

Example allowlist for JSON agents
---------------------------------

.. code-block:: python

   allowed_tools = (
       "fs.read_text",
       "text.word_count",
       "python.sandbox",
       "memory.search",
       "memory.write",
       "memory.stats",
       "eval.decision_matrix",
       "eval.pairwise_rank",
   )
