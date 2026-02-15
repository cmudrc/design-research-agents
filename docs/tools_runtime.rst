Tools Runtime
=============

``UnifiedToolRuntime`` is the canonical tools entrypoint. It merges tools from:

- Core tools (built-in, in-process)
- External MCP tools (stdio)
- Lazy tools (local scripts with docblock headers)

Use ``UnifiedToolRuntime`` constructor and classmethods for common setup.
Advanced parsing/types remain available under ``design_research_agents.tools.config``.

Quick start
-----------

.. code-block:: python

   from design_research_agents import UnifiedToolRuntime

   runtime = UnifiedToolRuntime()
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

Ergonomic setup (Python)
------------------------

.. code-block:: python

   import sys
   from design_research_agents import UnifiedToolRuntime
   from design_research_agents.tools.config import McpServerConfig

   runtime = UnifiedToolRuntime(
       workspace_root=".",
       enable_core_tools=True,
       lazy_search_paths=("examples/lazy_tools",),
       mcp_servers=(
           McpServerConfig(
               id="local_core",
               command=(sys.executable, "-m", "design_research_agents.mcp_server"),
               env={"PYTHONPATH": "src"},
           ),
       ),
   )

Focused presets:

.. code-block:: python

   from design_research_agents import UnifiedToolRuntime
   from design_research_agents.tools.config import McpServerConfig

   lazy_runtime = UnifiedToolRuntime.lazy(
       search_paths=("examples/lazy_tools",),
       workspace_root=".",
   )
   mcp_runtime = UnifiedToolRuntime.mcp(
       servers=(McpServerConfig(id="local_core", command=("python3", "-m", "design_research_agents.mcp_server")),),
       workspace_root=".",
   )

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

Load YAML config (advanced):

.. code-block:: python

   from design_research_agents import UnifiedToolRuntime

   runtime = UnifiedToolRuntime.from_yaml("tool_runtime.yaml")

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
