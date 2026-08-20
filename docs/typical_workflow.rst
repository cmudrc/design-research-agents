Typical Workflow
================

1. Choose inputs
----------------

Select a client/backend and define the task context you need to solve.

2. Instantiate core objects
---------------------------

Create tools, prompt templates, and either ``DirectLLMCall`` or
``MultiStepAgent``.

3. Execute or inspect
---------------------

Run one or more calls and inspect model output, tool calls, and step metadata.

4. Capture artifacts
--------------------

Persist execution outputs and traces for reproducible comparison.

5. Compose the ecosystem seams
------------------------------

Let `design-research-experiments
<https://cmudrc.github.io/design-research-experiments/>`_ bind agent runs to
tasks from `design-research-problems
<https://cmudrc.github.io/design-research-problems/>`_, then export records for
`design-research-analysis <https://cmudrc.github.io/design-research-analysis/>`_.
The experiments layer owns cross-package orchestration; this package owns agent
execution and normalization.

Choosing The Right Entry Point
------------------------------

.. list-table::
   :header-rows: 1

   * - Need
     - Recommended entry point
   * - Plain text generation
     - ``DirectLLMCall``
   * - Iterative reasoning without tools
     - ``MultiStepAgent(mode="direct")``
   * - Iterative tool use
     - ``MultiStepAgent(mode="json")``
   * - Code-action loops
     - ``MultiStepAgent(mode="code")``
   * - Higher-order orchestration
     - Patterns

Use ``DirectLLMCall`` when you want the shortest execution path and minimal
control overhead. Move to ``MultiStepAgent`` when interpretability and control
matter more than absolute latency. Move to patterns when you need coordinated
multi-role behavior and reusable orchestration logic.

If you are deciding between primitives, workflows, patterns, and exemplars,
see :doc:`where_to_start`.
