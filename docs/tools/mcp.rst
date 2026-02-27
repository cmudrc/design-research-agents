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

Troubleshooting
---------------

- ``Server '<id>' is not configured``: validate ``mcp.enabled`` and server id.
- ``Unknown MCP tool '<name>'``: inspect ``Toolbox.list_tools()`` for available names.
- Timeouts: increase ``timeout_s``.
- Missing env vars: set both ``env_allowlist`` and ``env`` entries.

Examples
--------

- ``examples/tools/mcp_minimal.py``
- ``examples/tools/source_fusion_story.py``
- ``examples/tools/README.md``
