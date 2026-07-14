design-research-agents
======================

A modular framework for building and studying AI agents in engineering design workflows.

What This Library Does
----------------------

``design-research-agents`` provides reusable abstractions for agent behavior,
tool use, workflow composition, and multi-step reasoning. It is built for
research workflows where traceability, reproducibility, and controlled
comparison matter as much as raw model capability.

Interpretable traces, explicit tool boundaries, and documented workflow
contracts are core features. They make agent studies easier to reproduce,
compare, and audit across experiments.

.. container:: drc-home-badges

   .. raw:: html

      <div class="drc-badge-row">
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml">
          <img alt="CI" src="https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg">
        </a>
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml">
          <img alt="Coverage" src="https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/coverage.svg">
        </a>
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml">
          <img alt="Examples Passing" src="https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/examples-passing.svg">
        </a>
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml">
          <img alt="API in Examples" src="https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/examples-api-coverage.svg">
        </a>
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml">
          <img alt="Docs" src="https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg">
        </a>
        <a class="drc-badge-link" href="https://pypi.org/project/design-research-agents/">
          <img alt="PyPI Version" src="https://img.shields.io/pypi/v/design-research-agents.svg">
        </a>
        <a class="drc-badge-link" href="https://pypi.org/project/design-research-agents/">
          <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/design-research-agents.svg">
        </a>
      </div>

Quality Signals
---------------

- ``Coverage`` reports total line coverage for the default deterministic test
  suite; CI requires at least 95%.
- ``Examples Passing`` reports checked-in example scripts that execute
  successfully in the examples workflow.
- ``API in Examples`` reports curated top-level ``__all__`` exports referenced
  by runnable examples. ``N/N`` means every supported top-level export appears
  in at least one example, and CI requires 100%.

Run ``make coverage``, ``make examples-test``, and ``make examples-coverage``
to reproduce these checks locally.

Highlights
----------

- Two core agent entry points: ``DirectLLMCall`` and ``MultiStepAgent``
- Explicit multi-step modes for ``direct``, ``json``, and ``code`` execution
- A study-facing execution facade in ``design_research_agents.study``
- Workflow primitives for model, tool, delegate, loop, and memory steps
- Model-selection policies with local and remote catalogs
- Tool contracts and schemas for safe, structured I/O
- Tracing hooks and emitters for debugging, evaluation, and reproducibility
- Workflow-native memory and reusable reasoning patterns including tree search,
  Ralph loops, nominal teams, debate, and RAG
- Runnable examples for deterministic validation and experimentation

The public surface is intentionally layered: start with ``DirectLLMCall`` for
one-shot execution, move to ``MultiStepAgent`` for managed loops, use
``Workflow`` when you need to author reusable graphs, reach for
``design_research_agents.patterns`` when a prebuilt orchestration strategy fits,
and use runnable examples as exemplars rather than as the primary abstraction.

Typical Workflow
----------------

1. Start from ``DirectLLMCall`` or ``MultiStepAgent`` depending on the level of
   control you need.
2. Configure runtime mode, tools, models, and any workflow or memory helpers.
3. Run a deterministic example or local quickstart to validate the environment.
4. Inspect traces, tool boundaries, and structured outputs for debugging and evaluation.
5. Reuse the same runtime contracts inside broader experiments through
   ``design_research_agents.study.AgentRunRequest``,
   ``design_research_agents.study.execute_agent_request(...)``,
   ``design_research_agents.study.normalize_agent_execution(...)``, and
   downstream analysis.

.. container:: drc-home-callout

   .. note::

      **New here?** Start with :doc:`where_to_start` to choose the right layer,
      then use :doc:`quickstart` to run the smallest end-to-end example.

Guides
------

Learn the base concepts, setup flow, and execution patterns that shape a stable
agent-research pipeline.

- :doc:`quickstart`
- :doc:`where_to_start`
- :doc:`installation`
- :doc:`vscode_setup`
- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`philosophy`

Examples
--------

Browse runnable examples and guided landing pages for the major public
subsystems.

- :doc:`examples/index`
- :doc:`agents/index`
- :doc:`llm_clients/index`
- :doc:`tools/index`
- :doc:`workflows/index`
- :doc:`patterns/index`

Reference
---------

Look up the stable import surface, package extras, and deeper API reference
material for the runtime boundaries that matter in CI and downstream studies.

- :doc:`api`
- :doc:`reference/index`
- :doc:`dependencies_and_extras`

Integration With The Ecosystem
------------------------------

The Design Research Collective maintains a modular ecosystem of libraries for
studying human and AI design behavior.

- **design-research-agents** implements AI participants, workflows, and tool-using reasoning patterns.
- **design-research-problems** provides benchmark design tasks, prompts, grammars, and evaluators.
- **design-research-analysis** analyzes the traces, event tables, and outcomes generated during studies.
- **design-research-experiments** sits above the stack as the study-design and orchestration layer, defining hypotheses, factors, conditions, replications, and artifact flows across agents, problems, and analysis.

Together these libraries support end-to-end design research pipelines, from
study design through execution and interpretation.

.. container:: drc-home-ecosystem

   .. image:: _static/ecosystem-platform.svg
      :alt: Ecosystem diagram showing experiments above agents, problems, and analysis.
      :class: dark-light drc-ecosystem-figure
      :width: 100%
      :align: center

Start Here
----------

- :doc:`quickstart`
- :doc:`where_to_start`
- :doc:`installation`
- :doc:`vscode_setup`
- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`examples/index`
- :doc:`api`
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/HEAD/CONTRIBUTING.md>`_

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   quickstart
   where_to_start
   installation
   vscode_setup
   concepts
   typical_workflow
   philosophy

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
   llm_clients/index
   tools/index
   agents/index
   workflows/index
   patterns/index
   reference/index
   dependencies_and_extras

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   documentation_automation
   CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/HEAD/CONTRIBUTING.md>
