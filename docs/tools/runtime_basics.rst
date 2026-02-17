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
       "calculator",
       {"expression": "12 * (4 + 1)"},
       request_id="example-tools-runtime",
       dependencies={},
   )

Built-in core tools
-------------------

- Math: ``calculator``
- Text: ``text.word_count``, ``text.extract_json``, ``text.diff``
- Filesystem: ``fs.list_dir``, ``fs.read_text``, ``fs.write_text``, ``fs.glob``,
  ``fs.stat``, ``fs.hash``
- Search/git: ``search.ripgrep``, ``git.status``, ``git.diff``, ``git.log``,
  ``git.show``
- Data/shell: ``data.load_csv``, ``data.describe``, ``bash.exec``

YAML config
-----------

.. code-block:: yaml

   core_tools:
     enabled: true
     workspace_root: .
     artifacts_dir: artifacts
     allow_network: false
     allow_writes_outside_artifacts: false
     allowed_commands: [git, rg, python, python3, uv, ruff, pytest]

   script_tools:
     enabled: true
     tools:
       - name: rubric_score
         path: examples/tools/script_tools/python/rubric_score.py
         description: Score text with a simple rubric.
         filesystem_write: true

   mcp:
     enabled: true
     servers:
       - id: local_core
         type: stdio
         command: [python3, -m, design_research_agents.mcp_server]
         timeout_s: 20

Examples
--------

- ``examples/tools/source_fusion_story.py``
- ``examples/workflow/workflow_schema_mode.py``
- ``examples/tools/README.md``
