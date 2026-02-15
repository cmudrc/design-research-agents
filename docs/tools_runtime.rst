Tools Runtime
=============

``UnifiedToolRuntime`` is the canonical tools entrypoint. It merges tools from:

- Core tools (built-in, in-process)
- External MCP tools (stdio)
- Lazy tools (local scripts with docblock headers)

Use ``dra.tools.UnifiedToolRuntime`` with ``dra.tools.ToolRuntimeConfig``
to control enabled sources and policy defaults.

Quick start
-----------

.. code-block:: python

   import design_research_agents as dra

   runtime = dra.tools.UnifiedToolRuntime()
   tools = runtime.list_tools()  # Sequence[ToolSpec]

   result = runtime.invoke(
       "calculator",
       {"expression": "12 * (4 + 1)"},
       request_id="example-tools-runtime",
       dependencies={},
   )

Tool names and routing
----------------------

- Core tools use plain names like ``calculator`` and ``fs.read_text``.
- Lazy tools are namespaced as ``lazy::<tool_name>``.
- MCP tools are namespaced as ``<server_id>::<tool_name>``.
- The runtime can resolve unqualified MCP names only when they are unique across
  all attached MCP servers.

Built-in core tools
-------------------

Math
^^^^

- ``calculator``

Text
^^^^

- ``text.word_count``
- ``text.extract_json``
- ``text.diff``

Filesystem
^^^^^^^^^^

- ``fs.list_dir``
- ``fs.read_text``
- ``fs.write_text``
- ``fs.glob``
- ``fs.stat``
- ``fs.hash``

Search and git
^^^^^^^^^^^^^^

- ``search.ripgrep``
- ``git.status``
- ``git.diff``
- ``git.log``
- ``git.show``

Data and shell
^^^^^^^^^^^^^^

- ``data.load_csv``
- ``data.describe``
- ``bash.exec`` (BashKit-backed; optional per-call command allowlist)

Bash allowlist control
----------------------

Use ``allowed_commands`` on each ``bash.exec`` call when you want the script
checked against a narrow command set:

.. code-block:: python

   result = runtime.invoke(
       "bash.exec",
       {
           "script": "git status --short",
           "allowed_commands": ["git"],
       },
       request_id="bash-allowlist-example",
       dependencies={},
   )

Runtime config (Python)
-----------------------

.. code-block:: python

   import sys
   import design_research_agents as dra

   config = dra.tools.ToolRuntimeConfig(
       core_tools=dra.tools.CoreToolsConfig(
           enabled=True,
           workspace_root=".",
           artifacts_dir="artifacts",
           allow_network=False,
       ),
       lazy_tools=dra.tools.LazyToolsConfig(
           enabled=True,
           search_paths=("examples/lazy_tools",),
       ),
       mcp=dra.tools.McpConfig(
           enabled=True,
           servers=(
               dra.tools.McpServerConfig(
                   id="local_core",
                   command=(sys.executable, "-m", "design_research_agents.mcp_server"),
                   env={"PYTHONPATH": "src"},
               ),
           ),
       ),
   )

   runtime = dra.tools.UnifiedToolRuntime(config=config)

Runtime config (YAML)
---------------------

.. code-block:: yaml

   core_tools:
     enabled: true
     workspace_root: .
     artifacts_dir: artifacts
     allow_network: false
     allow_writes_outside_artifacts: false
     allowed_commands: [git, rg, python, python3, uv, ruff, pytest]

   lazy_tools:
     enabled: true
     search_paths: [examples/lazy_tools]
     allow_network: false
     timeout_s_default: 30

   mcp:
     enabled: true
     servers:
       - id: local_core
         type: stdio
         command: [python3, -m, design_research_agents.mcp_server]
         timeout_s: 20
         env_allowlist: [PATH, HOME, USER, LANG, LC_ALL, PYTHONPATH, VIRTUAL_ENV]
         env:
           PYTHONPATH: src

Load YAML config:

.. code-block:: python

   import design_research_agents as dra

   config = dra.tools.load_tool_runtime_config("tool_runtime.yaml")
   runtime = dra.tools.UnifiedToolRuntime(config=config)

Related docs
------------

- :doc:`lazy_tools`
- :doc:`mcp`

Runnable examples
-----------------

.. code-block:: bash

   PYTHONPATH=src python3 examples/tools/source_fusion_story.py
   PYTHONPATH=src python3 examples/workflow/pure_tool_workflow.py

``examples/workflow/pure_tool_workflow.py`` initializes one workflow and
reuses ``run(inputs=...)`` for multiple runs over a caller-defined pure
tool/logic step graph.
