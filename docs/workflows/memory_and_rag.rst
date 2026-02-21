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

Background references
---------------------

- `Retrieval-Augmented Generation (RAG) <https://arxiv.org/abs/2005.11401>`_

The original RAG formulation combines a retriever and generator with
non-parametric memory; this project implements a workflow-level retrieval and
context-injection pattern.

Example
-------

See ``examples/workflow/rag_reasoning.py`` and
``examples/agents/basic/multi_step_json_with_memory.py``.
