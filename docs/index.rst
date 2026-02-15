design-research-agents
======================

A flexible, modular framework for researching AI agents that design

Build and compare agent behaviors, swap LLM backends, and capture traces
without rewriting your pipeline. The library favors small, composable
pieces so you can test ideas quickly and keep experiments reproducible.

Highlights
----------

- Multiple agent styles: direct LLM, tool-calling, router, and multi-step.
- Model selection policies with local/remote catalogs.
- Tool contracts and schemas for safe, structured I/O.
- Tracing hooks and emitters for debugging and evaluation.
- Streaming examples for real-time UX and analysis.

Typical workflow
----------------

1. Choose an agent type and backend.
2. Define tools, prompts, and policies.
3. Run experiments and capture traces.
4. Compare results and iterate.

Get started
-----------

- :doc:`quickstart` for a fast, end-to-end example.
- :doc:`agent_types` to understand the tradeoffs.
- :doc:`api` for reference details.
- :doc:`contributing` if you want to extend the project.

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   quickstart
   philosophy
   agent_types
   tools_runtime
   mcp
   lazy_tools
   contributing

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
