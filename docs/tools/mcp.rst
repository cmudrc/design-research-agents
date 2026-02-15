MCP Tools
=========

The package includes a built-in stdio MCP server and MCP client integration in
``Toolbox``.

Server
------

Run the built-in server:

.. code-block:: bash

   dra mcp serve

Integration steps
-----------------

1. Confirm target server supports stdio MCP tool methods.
2. Add a server entry under ``mcp.servers``.
3. Use a unique ``id``; that becomes tool namespace prefix.
4. Verify with ``dra mcp ping``.
5. Invoke as ``<id>::<tool_name>``.

CLI helpers
-----------

.. code-block:: bash

   dra mcp ping --server <id> --config tool_runtime.yaml
   dra mcp call <tool_name> --json '{"arg":"value"}' --config tool_runtime.yaml

Troubleshooting
---------------

- ``Server '<id>' is not configured``: validate ``mcp.enabled`` and server id.
- ``Unknown MCP tool '<name>'``: run ``dra mcp ping --server <id>``.
- Timeouts: increase ``timeout_s``.
- Missing env vars: set both ``env_allowlist`` and ``env`` entries.

Examples
--------

- ``examples/tools/mcp_minimal.py``
- ``examples/tools/source_fusion_story.py``
- ``examples/tools/README.md``
