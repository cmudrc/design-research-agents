Runtime Basics
==============

Use ``Toolbox`` for a single invocation surface across tool sources.

Quick start
-----------

.. code-block:: python

   from design_research_agents import Toolbox

   runtime = Toolbox()
   tools = runtime.list_tools()
   result = runtime.invoke(
       "text.word_count",
       {"text": "design research agents"},
       request_id="example-tools-runtime",
       dependencies={},
   )

Built-in core tools
-------------------

- Python: ``python.sandbox``
- Text: ``text.word_count``, ``text.extract_json``, ``text.diff``
- Filesystem: ``fs.list_dir``, ``fs.read_text``, ``fs.write_text``, ``fs.glob``,
  ``fs.stat``, ``fs.hash``
- Search/git: ``search.ripgrep``, ``git.status``, ``git.diff``, ``git.log``,
  ``git.show``
- Data/shell: ``data.load_csv``, ``data.describe``, ``bash.exec``
- Memory: ``memory.search``, ``memory.write``, ``memory.stats``
- Evaluation: ``eval.decision_matrix``, ``eval.pairwise_rank``

Network-gated tools
--------------------

Some core tools require outbound network access and are not registered by
default. Pass ``allow_network=True`` to ``Toolbox`` to opt in:

.. code-block:: python

   from design_research_agents import Toolbox

   runtime = Toolbox(allow_network=True)
   result = runtime.invoke_dict(
       "web.instant_answer",
       {"query": "Carnegie Mellon University"},
       request_id="example-instant-answer",
       dependencies={},
   )

- ``web.instant_answer``: quick factual or encyclopedic lookup using a
  no-key-required instant-answer provider. This is **not** general web
  search — it returns a topic summary and related-topic titles, URLs, and
  snippets for well-known entities, and frequently returns no results for
  open-ended research or discovery queries.
- ``web.search``: ranked, relevance-scored general web search using the
  `Tavily Search API <https://tavily.com>`_. Suitable for open-ended research
  and discovery queries, unlike ``web.instant_answer``. Requires a
  ``TAVILY_API_KEY`` environment variable; the tool is silently omitted from
  ``list_tools()`` when that variable is unset or blank, rather than being
  registered and always failing at invocation time. Tavily offers a free
  tier (no credit card required) for obtaining a key.

When ``allow_network`` is left at its default ``False``, network-gated tools
are omitted from ``list_tools()`` entirely rather than being present but
rejecting calls, so callers do not need to catch a network-disabled error to
discover this. The same applies to ``web.search`` specifically when
``TAVILY_API_KEY`` is not configured, even with ``allow_network=True``.

Examples
--------

- ``examples/tools/multi_source_tool_usage.py``
- ``examples/workflow/workflow_schema_mode.py``
- ``examples/tools/README.md``
