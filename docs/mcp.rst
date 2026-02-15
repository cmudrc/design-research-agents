MCP
===

The package includes a built-in stdio MCP server and a stdio MCP client source.

Server
------

Run the built-in server:

.. code-block:: bash

   dra mcp serve

Client
------

Attach external MCP servers in runtime config and call tools through the unified
runtime. MCP tools are namespaced as ``<server_id>::<tool_name>``.

CLI helpers:

.. code-block:: bash

   dra mcp ping --server <id> --config tool_runtime.yaml
   dra mcp call <tool_name> --json '{"arg":"value"}' --config tool_runtime.yaml
