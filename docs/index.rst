design-research-agents
======================

A flexible, modular framework for researching AI agents for design workflows.

Build and compare agent behaviors, swap LLM backends, and capture traces
without rewriting your pipeline. The library favors small, composable
pieces so you can test ideas quickly and keep experiments reproducible.

Highlights
----------

- Eight core agent styles: ``SingleStepDirectLLMAgent``, ``SingleStepToolRouterAgent``,
  ``SingleStepJsonToolCallingAgent``, ``SingleStepCodeToolCallingAgent``,
  ``MultiStepDirectLLMAgent``, ``MultiStepToolRouterAgent``,
  ``MultiStepJsonToolCallingAgent``, and ``MultiStepCodeToolCallingAgent``.
- Model selection policies with local/remote catalogs.
- Tool contracts and schemas for safe, structured I/O.
- Tracing hooks and emitters for debugging and evaluation.
- Streaming examples for real-time UX and analysis.
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
