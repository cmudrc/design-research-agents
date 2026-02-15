Tools Runtime
=============

The framework now ships a unified tool runtime that can merge multiple sources:

- Core tools (built-in, in-process)
- External MCP tools (stdio)
- Lazy tools (local scripts with docblock headers)

Use ``design_research_agents.tools.UnifiedToolRuntime`` with
``ToolRuntimeConfig`` to control enabled sources and policy defaults.

Example:

.. code-block:: python

   from design_research_agents.tools import ToolRuntimeConfig, UnifiedToolRuntime

   runtime = UnifiedToolRuntime(config=ToolRuntimeConfig())
   tools = runtime.list_tools()

Core tools include filesystem, search, git, text utilities, ``run.command``, and
``bash.exec`` (backed by BashKit).
