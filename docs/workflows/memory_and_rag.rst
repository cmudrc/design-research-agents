Memory Primitives and RAG
=========================

This project now supports workflow-first memory primitives:

- ``MemoryReadStep``
- ``MemoryWriteStep``

And a local pluggable persistent memory store:

- ``SQLiteMemoryStore`` (default path:
  ``artifacts/memory/memory.sqlite3``)

Memory retrieval
----------------

``SQLiteMemoryStore.search(...)`` always computes lexical relevance.
When embeddings are available, the final ranking score uses:

- ``0.7 * vector_score + 0.3 * lexical_score``

If embeddings are unavailable, retrieval falls back to lexical-only ranking.

RAG orchestration
-----------------

``RagReasoningPattern`` composes memory and reasoning as:

1. ``MemoryReadStep``
2. ``AgentStep`` reasoning delegate (with retrieved context injection)
3. ``MemoryWriteStep`` (optional write-back)

Example
-------

See ``examples/workflow/rag_reasoning.py`` and
``examples/agents/basic/multi_step_json_with_memory.py``.
