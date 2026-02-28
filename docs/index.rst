design-research-agents
======================

A flexible, modular framework for researching AI agents in design workflows.

Build and compare agent behaviors, swap LLM backends, and capture traces
without rewriting your pipeline. The library favors small, composable
pieces so you can test ideas quickly and keep experiments reproducible.

Highlights
----------

- Two core agent entry points: ``DirectLLMCall`` and ``MultiStepAgent``.
- ``MultiStepAgent`` supports explicit modes: ``direct``, ``json``, and ``code``.
- JSON mode uses structured ``tool_name``/``tool_input`` selection for
  iterative tool-call loops.
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
- :doc:`dependencies_and_extras` for optional dependency profiles and platform constraints.
- :doc:`examples/index` for scenario-driven runnable examples and expected observations.
- :doc:`examples/workflow/index` for runnable workflow primitive examples.
- :doc:`examples/patterns/index` for runnable orchestration pattern examples.
- :doc:`llm_clients/index` to choose local or remote client backends.
- :doc:`tools/index` for unified runtime + MCP + script tools.
- :doc:`agents/index` to understand agent execution tradeoffs.
- :doc:`workflows/index` for workflow builder primitives and composition.
- :doc:`patterns/index` for prebuilt workflow implementations.
- :doc:`api` for the guaranteed public API surface.
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md>`_
  for contribution workflow and PR expectations.

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   quickstart
   dependencies_and_extras
   examples/index
   philosophy
   llm_clients/index
   tools/index
   agents/index
   workflows/index
   patterns/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
   reference/index
