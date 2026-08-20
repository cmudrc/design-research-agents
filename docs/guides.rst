Guides
======

``design-research-agents`` is the execution layer of the
CMU Design Research Collective design-research ecosystem. It owns executable
participants, workflow and tool runtimes, model-client adapters, and traceable
run results.

Primary Path
------------

Follow the same path used across the design-research libraries:

1. :doc:`installation` — create a supported Python environment and choose extras.
2. :doc:`quickstart` — run the offline base-install example.
3. :doc:`concepts` and :doc:`typical_workflow` — learn the contracts and execution flow.
4. :doc:`examples/index` — move from the base example to realistic participants and backends.
5. :doc:`api` — confirm the compatibility-guaranteed import surface.

Additional Guides
-----------------

- :doc:`where_to_start` helps choose among agents, workflows, patterns, and examples.
- :doc:`vscode_setup` provides an editor-first setup and debugging path.
- :doc:`philosophy` explains the package's research-oriented design choices.
- :doc:`dependencies_and_extras` maps optional backends to install profiles.

Compatibility and Ecosystem
---------------------------

The curated top-level API and documented public facades are the compatibility
boundary. ``design_research_agents.study`` is preferred for new study runners;
``design_research_agents.integration`` remains available for compatibility.
See :doc:`api` and :doc:`reference/index` for the exact distinction.

Sibling layers own adjacent responsibilities:

- `design-research-problems <https://cmudrc.github.io/design-research-problems/>`_ owns tasks and evaluators.
- `design-research-experiments <https://cmudrc.github.io/design-research-experiments/>`_ owns study design and orchestration.
- `design-research-analysis <https://cmudrc.github.io/design-research-analysis/>`_ owns validation and downstream analysis.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   quickstart
   concepts
   typical_workflow
   where_to_start
   vscode_setup
   philosophy
   dependencies_and_extras
