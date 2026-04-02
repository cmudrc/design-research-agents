Where To Start
==============

``design-research-agents`` has a deliberate abstraction ladder. Start with the
smallest layer that matches your need, then move upward only when the lower
layer becomes constraining.

Abstraction Ladder
------------------

.. list-table::
   :header-rows: 1

   * - Layer
     - Start here when
     - Primary surface
   * - Primitive direct call
     - You want one prompt in and one structured result out with minimal control overhead.
     - ``DirectLLMCall``
   * - Managed multi-step agent
     - You need iterative reasoning, tool use, or code-action loops but still want a packaged agent entry point.
     - ``MultiStepAgent``
   * - Reusable workflow graph
     - You want to author topology directly from typed steps and reuse that graph across runs.
     - ``Workflow``
   * - Prebuilt orchestration strategy
     - A standard coordination form already matches the study design you have in mind.
     - ``design_research_agents.patterns``
   * - Runnable exemplar
     - You want a copyable reference implementation, expected outputs, and dependency notes.
     - :doc:`examples/index`

Quick Chooser
-------------

- Start with ``DirectLLMCall`` when latency and a small surface matter more than orchestration control.
- Start with ``MultiStepAgent`` when you want managed iterative behavior without authoring a custom graph.
- Start with ``Workflow`` when the graph itself is the thing you are designing and validating.
- Start with :doc:`patterns/index` when you want a reusable plan/execute, debate, routing, RAG, or related pattern.
- Start with :doc:`examples/index` when you want exemplars to copy from, not a new abstraction to extend.

Specialized Public Entrypoints
------------------------------

Two public entry points sit beside the main ladder because they target packaged
problem studies rather than general prompting:

- ``SeededRandomBaselineAgent`` is the lightweight control-condition participant.
- ``PromptWorkflowAgent`` wraps a prompt-mode ``Workflow`` when experiment code should own the problem object and run metadata.

Recommended First Click
-----------------------

- New to the library entirely: :doc:`quickstart`
- Choosing between agent entry points: :doc:`agents/index`
- Building your own reusable topology: :doc:`workflows/index`
- Reusing a higher-order orchestration strategy: :doc:`patterns/index`
- Looking for copyable end-to-end examples: :doc:`examples/index`
