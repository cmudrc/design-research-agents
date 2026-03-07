design-research-agents
======================

A modular framework for engineering design agent research and experimentation.

Use it to:

- prototype direct and multi-step agent behavior,
- orchestrate workflows with explicit runtime steps and tools, and
- compare backends and orchestration patterns with reproducible traces.

Highlights
----------

- Two core agent entry points: ``DirectLLMCall`` and ``MultiStepAgent`` (``direct``, ``json``, ``code`` modes).
- A workflow runtime with explicit primitives for model, tool, delegate, loop, and memory steps.
- A unified tool runtime via ``Toolbox`` for callable, script, and MCP-backed sources.
- Hosted and local LLM clients, plus ``ModelSelector`` policy-driven backend selection.
- Prebuilt orchestration patterns for plan/execute, propose/critic, debate, routing, beam search, RAG, blackboard, and conversation.
- Tracing hooks and structured ``ExecutionResult`` outputs for repeatable evaluation.

Typical Workflow
----------------

1. Choose an agent entry point and backend strategy.
2. Configure tools, prompts, and execution policies.
3. Run experiments and capture traces/artifacts.
4. Compare outcomes and iterate on workflow design.

Start Here
----------

- :doc:`quickstart` for a fast, end-to-end example.
- :doc:`dependencies_and_extras` for install profiles, extras, and release checks.
- :doc:`examples/index` for scenario-driven runnable examples.
- :doc:`examples/workflow/index` for runnable workflow primitive examples.
- :doc:`examples/patterns/index` for runnable orchestration pattern examples.
- :doc:`llm_clients/index` to choose local or remote LLM client backends.
- :doc:`tools/index` for runtime basics plus script and MCP tools.
- :doc:`agents/index` for agent execution tradeoffs.
- :doc:`workflows/index` for workflow-builder primitives and composition.
- :doc:`patterns/index` for reusable workflow implementations.
- :doc:`api` for the curated public API surface.
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md>`_
  for contribution workflow and quality gates.

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
