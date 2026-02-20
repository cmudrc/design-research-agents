design-research-agents
======================

A flexible, modular framework for researching AI agents for design workflows.

Build and compare agent behaviors, swap LLM backends, and capture traces
without rewriting your pipeline. The library favors small, composable
pieces so you can test ideas quickly and keep experiments reproducible.

Highlights
----------

- Two core agent entry points: ``DirectLLMCall`` and ``MultiStepAgent``.
- ``MultiStepAgent`` supports explicit modes: ``direct``, ``json``, and ``code``.
- JSON mode automatically uses a TOOL_CALL/STOP router schema when all tools
  are arg-less (no structured input fields).
- Model selection policies with local/remote catalogs.
- Tool contracts and schemas for safe, structured I/O.
- Tracing hooks and emitters for debugging and evaluation.
- Runnable examples for deterministic validation and experimentation.
- Workflow-native memory, networked blackboard coordination, and reusable
  reasoning patterns (tree search and RAG).

Typical workflow
----------------

1. Choose an agent type and backend.
2. Define tools, prompts, and policies.
3. Run experiments and capture traces.
4. Compare results and iterate.

Get started
-----------

- :doc:`quickstart` for a fast, end-to-end example.
- :doc:`llm_clients/index` to choose local or remote client backends.
- :doc:`tools/index` for unified runtime + MCP + script tools.
- :doc:`agents/index` to understand agent execution tradeoffs.
- :doc:`workflows/index` for orchestration patterns and composition.
- :doc:`api` for reference details.
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md>`_
  for contribution workflow and PR expectations.

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   quickstart
   philosophy
   llm_clients/index
   tools/index
   agents/index
   workflows/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
   reference/index
