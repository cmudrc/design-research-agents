Tools
=====

``Toolbox`` is the canonical tools entrypoint. It fuses core tools,
MCP tools, and script tools into one runtime surface.

Naming and routing
------------------

- Core tools use plain names (for example ``calculator`` or ``fs.read_text``).
- Script tools are namespaced as ``script::<tool_name>``.
- MCP tools are namespaced as ``<server_id>::<tool_name>``.

Pages
-----

- :doc:`/examples/tools_and_integrations`
- :doc:`runtime_basics`
- :doc:`design_stem_tools`
- :doc:`mcp`
- :doc:`script_tools`

.. toctree::
   :maxdepth: 2
   :hidden:

   runtime_basics
   design_stem_tools
   mcp
   script_tools
