design-research-agents
======================

A modular framework for building and studying AI agents in engineering design workflows.

What This Library Does
----------------------

``design-research-agents`` provides reusable abstractions for agent behavior,
tool use, workflow composition, and multi-step reasoning. We use it in research
settings where traceability, reproducibility, and experimental control matter as
much as raw model capability.

Highlights
----------

- Agent abstractions
- Tool-use runtime
- Workflow primitives
- Orchestration patterns
- Trace capture
- Backend flexibility

This library is not only for deploying agents. It is designed to support agent
**study**: reproducible runs, behavioral comparison, and interpretable execution
artifacts that can be analyzed across controlled conditions.

Typical Workflow
----------------

1. Choose an LLM client and execution setting.
2. Define tools, prompts, and constraints.
3. Select ``DirectLLMCall``, ``MultiStepAgent``, or a higher-order pattern.
4. Execute runs and capture traces.
5. Export outputs for downstream experiment orchestration and analysis.

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

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   quickstart
   installation
   concepts
   typical_workflow
   examples/index
   api

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   dependencies_and_extras
   Contributing <https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md>

.. toctree::
   :maxdepth: 2
   :caption: Additional Guides
   :hidden:

   llm_clients/index
   tools/index
   agents/index
   workflows/index
   patterns/index
   reference/index
   philosophy
