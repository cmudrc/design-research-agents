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

Integrate another MCP server
----------------------------

1. Confirm the MCP server supports stdio transport and the MCP tool methods.
2. Add a server entry under ``mcp.servers`` in your runtime config.
3. Start with a unique ``id``; that id becomes the tool namespace prefix.
4. Verify integration with ``dra mcp ping``.
5. Invoke tools via ``<id>::<tool_name>``.

YAML config example:

.. code-block:: yaml

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
       - id: design_ops
         type: stdio
         command: [uv, run, design-ops-mcp]
         timeout_s: 30
         env_allowlist: [PATH, HOME, USER, LANG, LC_ALL, DESIGN_OPS_API_KEY]
         env:
           DESIGN_OPS_API_KEY: ${DESIGN_OPS_API_KEY}

Python config example:

.. code-block:: python

   from design_research_agents.tools import ToolRuntimeConfig, UnifiedToolRuntime
   from design_research_agents.tools.config import McpConfig, McpServerConfig

   config = ToolRuntimeConfig(
       mcp=McpConfig(
           enabled=True,
           servers=(
               McpServerConfig(
                   id="design_ops",
                   command=("uv", "run", "design-ops-mcp"),
                   timeout_s=30,
                   env_allowlist=("PATH", "HOME", "USER", "LANG", "LC_ALL", "DESIGN_OPS_API_KEY"),
                   env={"DESIGN_OPS_API_KEY": "set-at-runtime"},
               ),
           ),
       )
   )
   runtime = UnifiedToolRuntime(config=config)

Tool naming and alias behavior
------------------------------

- Canonical MCP tool name is always ``<server_id>::<tool_name>``.
- Unqualified invocation (for example ``calculator``) works only when exactly
  one MCP server exposes that tool name and no core/lazy tool conflicts.
- For predictable behavior in multi-server setups, always use namespaced names.

CLI helpers:

.. code-block:: bash

   dra mcp ping --server <id> --config tool_runtime.yaml
   dra mcp call <tool_name> --json '{"arg":"value"}' --config tool_runtime.yaml

Programmatic invocation example:

.. code-block:: python

   result = runtime.invoke(
       "design_ops::research.summary",
       {"topic": "agent evals", "days": 7},
       request_id="mcp-example",
       dependencies={},
   )

Troubleshooting
---------------

- ``Server '<id>' is not configured``:
  - Check ``mcp.enabled`` and the server ``id`` in YAML.
- ``Unknown MCP tool '<name>'``:
  - Run ``dra mcp ping --server <id>`` to confirm discovered tool names.
- Timeout failures:
  - Increase ``timeout_s`` for slow MCP servers.
- Missing environment variables:
  - Include variable names in ``env_allowlist`` and values in ``env``.

Runnable examples:

.. code-block:: bash

   PYTHONPATH=src python3 examples/tools/mcp_minimal.py
   PYTHONPATH=src python3 examples/tools/source_fusion_story.py
