Workflows
=========

Workflow utilities provide the builder surface for authoring new workflow
implementations from reusable step primitives.

Quick chooser
-------------

- Need a reusable constructor-first step graph: start with ``Workflow``.
- Need looped orchestration inside a custom graph: use ``LoopStep``.
- Need deterministic local transforms and branching: use ``LogicStep``.
- Need explicit tool boundaries: use ``ToolStep``.
- Need delegated nested execution: use ``AgentStep`` or ``DelegateBatchStep``.
- Need a prebuilt implementation (debate, routing, plan-execute, tree search, RAG):
  use :doc:`/patterns/index`.

Pages
-----

- :doc:`/examples/workflow/index`
- :doc:`runtime_and_steps`
- :doc:`composition_guide`
- :doc:`architecture_boundaries`

.. toctree::
   :maxdepth: 2
   :hidden:

   runtime_and_steps
   composition_guide
   architecture_boundaries
