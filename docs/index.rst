design-research-agents
======================

The agent-execution layer for reproducible design research.

What This Library Does
----------------------

``design-research-agents`` owns executable AI participants, workflow and tool
runtimes, model-client adapters, and traceable run results. It is built for
research workflows where reproducibility and controlled comparison matter as
much as raw model capability.

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

      **New here?** Follow :doc:`guides` for the shared install → quickstart →
      concepts/workflow → examples → API path. The base quickstart is offline
      and requires no model service.

Guides
------

Learn the base concepts, setup flow, and execution patterns that shape a stable
agent-research pipeline.

- :doc:`guides`
- :doc:`installation`
- :doc:`quickstart`
- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`where_to_start`

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

- :doc:`reference/index`

Integration With The Ecosystem
------------------------------

The CMU Design Research Collective design-research ecosystem is a modular set of
libraries for studying human and AI design behavior.

- **design-research-agents** (this package) executes AI participants, workflows, and tool-using reasoning patterns.
- `design-research-problems <https://cmudrc.github.io/design-research-problems/>`_ owns benchmark design tasks, prompts, grammars, and evaluators.
- `design-research-experiments <https://cmudrc.github.io/design-research-experiments/>`_
  owns study design and coordinates artifact flows across packages.
- `design-research-analysis <https://cmudrc.github.io/design-research-analysis/>`_ validates and analyzes the resulting traces, event tables, and outcomes.

Together these libraries support end-to-end design research pipelines, from
study design through execution and interpretation.

The figure shows two complementary views: control responsibility and runtime
artifact flow. Neither view is a package-install order. See the umbrella
`compatibility matrix <https://cmudrc.github.io/design-research/compatibility.html>`_
for the component versions tested together.

.. container:: drc-home-ecosystem

   .. image:: _static/ecosystem-platform.svg
      :alt: Two-view diagram showing the control topology and runtime data flow across Problems, Agents, Experiments, and Analysis.
      :class: dark-light drc-ecosystem-figure
      :width: 100%
      :align: center

Start Here
----------

- :doc:`installation`
- :doc:`quickstart`
- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`examples/index`
- :doc:`api`
- :doc:`guides`
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/HEAD/CONTRIBUTING.md>`_

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   guides

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Subsystems
   :hidden:

   llm_clients/index
   tools/index
   agents/index
   workflows/index
   patterns/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   documentation_automation
   CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/HEAD/CONTRIBUTING.md>
