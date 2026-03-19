Memory and RAG Pattern
======================

Workflow-native memory primitives are available in
``design_research_agents.workflow``:

- ``MemoryReadStep``
- ``MemoryWriteStep``

And a local pluggable persistent memory store:

- ``SQLiteMemoryStore`` (default path:
  ``artifacts/memory/memory.sqlite3``)
- ``ChromaMemoryStore`` (optional ``memory_chroma`` extra)

Graph memory backends are also available for relational design context:

- ``NetworkXGraphMemoryStore`` (optional ``memory_graph`` extra)

Memory retrieval
----------------

``SQLiteMemoryStore.search(...)`` always computes lexical relevance.
When embeddings are available, the final ranking score uses:

- ``0.7 * vector_score + 0.3 * lexical_score``

If embeddings are unavailable, retrieval falls back to lexical-only ranking.

Built-in knowledge profiles
---------------------------

The public ``design_research_agents.memory`` facade also exposes deterministic
built-in knowledge profiles for ``stem``, ``aerospace``, and ``mechanics``.
Use ``seed_builtin_knowledge_profile(...)`` to preload a text/vector store,
a graph store, or both.

For example, this seeds a mechanics baseline into a SQLite store:

.. code-block:: python

   from design_research_agents.memory import SQLiteMemoryStore, seed_builtin_knowledge_profile

   with SQLiteMemoryStore() as store:
       seed_builtin_knowledge_profile("mechanics", memory_store=store, namespace="design")

The reproducible materialization helper script can write the checked-in profile
data into local artifacts:

.. code-block:: bash

   PYTHONPATH=src python scripts/build_knowledge_profile.py \
     --profile mechanics \
     --sqlite-db artifacts/memory/mechanics.sqlite3 \
     --graph-json artifacts/memory/mechanics_graph.json \
     --summary-json artifacts/memory/mechanics_summary.json

RAG orchestration
-----------------

``RAGPattern`` composes memory and reasoning as:

1. ``MemoryReadStep``
2. ``LogicStep`` graph retrieval (optional)
3. ``DelegateStep`` reasoning delegate (with retrieved text and graph context injection)
4. ``MemoryWriteStep`` (optional write-back)

Background reference
--------------------

- `Retrieval-Augmented Generation (RAG) <https://arxiv.org/abs/2005.11401>`_

The original RAG formulation combines a retriever and generator with
non-parametric memory; this implementation uses workflow-level retrieval and
context injection.

Examples
--------

- ``examples/patterns/rag.py``
- ``examples/agents/multi_step_json_with_memory.py``
