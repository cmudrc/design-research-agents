Tools
=====

``Toolbox`` is the canonical tool runtime for ``design-research-agents``. It
combines core tools, script tools, and MCP-backed tools behind one invocation
surface.

Tool Sources
------------

- **Core tools**: packaged utilities for text, filesystem, search, shell, data,
  memory, and evaluation tasks.
- **Script tools**: repository-local scripts exposed through explicit
  ``ScriptToolConfig`` registration.
- **MCP tools**: external tools discovered from configured MCP servers.

Namespaces
----------

- Core tools: ``<domain>.<name>`` (for example ``text.word_count``)
- Script tools: ``script::<name>``
- MCP tools: ``<server_id>::<tool_name>``

Runtime Selection Behavior
--------------------------

Selection is explicit at runtime: the invoked name determines source and
execution policy. This keeps tool provenance auditable and avoids hidden routing
state. In agent loops, tool schemas define what is callable and how model output
is validated before invocation.

Authoring and Registration
--------------------------

1. Define tool behavior (core callable, script entrypoint, or MCP server endpoint).
2. Register in ``Toolbox`` via ``CallableToolConfig``, ``ScriptToolConfig``, or
   ``MCPServerConfig``.
3. Inspect exposed specs with ``Toolbox.list_tools()``.
4. Invoke through ``Toolbox.invoke(...)`` and persist structured results.

Compact Chooser
---------------

.. list-table::
   :header-rows: 1

   * - Need
     - Recommended tool source
   * - Pure Python utility in the same process
     - Core callable tool
   * - Reuse existing script with stable I/O
     - Script tool
   * - Integrate external capability/service
     - MCP tool

Core callables are simplest for deterministic in-process operations. Script
tools are useful when you already have executable scripts or want strict input
and output envelopes without embedding all logic in Python modules. MCP tools
are best when capabilities live outside the process boundary and need explicit
server lifecycle management.

Pages
-----

- :doc:`runtime_basics`
- :doc:`design_research_tools`
- :doc:`script_tools`
- :doc:`mcp`
- :doc:`/examples/tools/index`

.. toctree::
   :maxdepth: 2
   :hidden:

   runtime_basics
   design_research_tools
   script_tools
   mcp
