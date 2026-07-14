MCP Tools
=========

The package includes MCP client integration in ``Toolbox``.

Server
------

Use a stdio MCP server command from your selected provider/runtime.

.. code-block:: bash

   python3 -m your_mcp_server_module

Integration steps
-----------------

1. Confirm target server supports stdio MCP tool methods.
2. Add a server entry under ``mcp.servers``.
3. Use a unique ``id``; that becomes tool namespace prefix.
4. Verify tools are exposed via ``Toolbox.list_tools()``.
5. Invoke as ``<id>::<tool_name>``.

Programmatic helpers
--------------------

.. code-block:: python

   from design_research_agents import MCPServerConfig, Toolbox

   runtime = Toolbox(
       mcp_servers=(
           MCPServerConfig(
               id="local_core",
               command=("python3", "-m", "your_mcp_server_module"),
           ),
       )
   )
   names = [spec.name for spec in runtime.list_tools() if spec.name.startswith("local_core::")]
   result = runtime.invoke(
       "local_core::text.word_count",
       {"text": "design research"},
       request_id="docs-mcp",
       dependencies={},
   )
   runtime.close()

For Python module-backed servers, use ``MCPServerConfig.python_module(...)`` to
avoid hand-building the ``python -m`` command.

.. code-block:: python

   from design_research_agents import MCPServerConfig, Toolbox

   runtime = Toolbox(
       enable_core_tools=False,
       mcp_servers=(
           MCPServerConfig.python_module(
               id="drp_problem",
               module="design_research_problems.mcp",
               args=("pill_capsule_min_area", "--no-citation"),
           ),
       ),
   )
   tools = [spec.name for spec in runtime.list_tools()]
   runtime.close()

For packaged ``design-research-problems`` tasks, launch DERP's maintained MCP
entrypoint directly. This avoids temporary server scripts and keeps problem
briefs, evaluation tools, and solver hints on the library-owned path.

.. code-block:: bash

   python -m pip install "design-research-agents[mcp,gemini]" "design-research-problems[mcp]"

.. code-block:: python

   from design_research_agents import MCPServerConfig, Toolbox

   runtime = Toolbox(
       enable_core_tools=False,
       mcp_servers=(
           MCPServerConfig.python_module(
               id="drp_problem",
               module="design_research_problems.mcp",
               args=("pill_capsule_min_area", "--no-citation"),
               timeout_s=45,
           ),
       ),
   )
   hints = runtime.invoke(
       "drp_problem::solver_hints",
       {},
       request_id="docs-derp-hints",
       dependencies={},
   )
   runtime.close()

Troubleshooting
---------------

- ``Server '<id>' is not configured``: validate ``mcp.enabled`` and server id.
- ``Unknown MCP tool '<name>'``: inspect ``Toolbox.list_tools()`` for available names.
- Timeouts: the error includes ``timeout_s`` and the stdio command. Increase
  ``MCPServerConfig.timeout_s`` for genuinely long-running tools; otherwise
  verify the command and captured stderr.
- Missing env vars: set both ``env_allowlist`` and ``env`` entries.

Examples
--------

- ``examples/tools/mcp_minimal.py``
- ``examples/tools/multi_source_tool_usage.py``
- ``examples/tools/derp_mcp_general_solver.py``
- ``examples/tools/README.md``
