design-research-agents
======================

A modular framework for building and studying AI agents in engineering design workflows.

``design-research-agents`` provides reusable abstractions for agent behavior,
tool use, workflow composition, and multi-step reasoning. It is built for
research workflows where traceability, reproducibility, and controlled
comparison matter as much as raw model capability.

Interpretable traces, explicit tool boundaries, and documented workflow
contracts are core features. They make agent studies easier to reproduce,
compare, and audit across experiments.

.. raw:: html

   <div class="drc-badge-row">
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml">
       <img alt="CI" src="https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg">
     </a>
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml">
       <img alt="Coverage" src="https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/coverage.svg">
     </a>
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml">
       <img alt="Examples Passing" src="https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/examples-passing.svg">
     </a>
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml">
       <img alt="Public API In Examples" src="https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/examples-api-coverage.svg">
     </a>
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml">
       <img alt="Docs" src="https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg">
     </a>
   </div>

.. note::

   **Start with** :doc:`quickstart` to run a first agent workflow, inspect the
   public runtime surface, and get the package into a stable local loop before
   diving into the broader patterns and reference material.

Guides
------

Learn the base concepts, setup flow, and execution patterns that shape a stable
agent-research pipeline.

- :doc:`quickstart`
- :doc:`installation`
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

.. image:: _static/ecosystem-platform.svg
   :alt: Ecosystem diagram showing experiments above agents, problems, and analysis.
   :width: 100%
   :align: center

Start Here
----------

- :doc:`quickstart`
- :doc:`installation`
- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`examples/index`
- :doc:`api`
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md>`_

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   quickstart
   installation
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

   CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md>
