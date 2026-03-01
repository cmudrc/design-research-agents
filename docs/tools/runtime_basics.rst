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

Examples
--------

- ``examples/tools/multi_source_tool_usage.py``
- ``examples/workflow/workflow_schema_mode.py``
- ``examples/tools/README.md``
