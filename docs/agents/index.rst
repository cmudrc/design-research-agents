Agents
======

The framework provides four public concrete agent entry points. All four use
the same public delegate contract: ``compile(prompt, ...)`` for workflow
construction and ``run(prompt, ...)`` for execution. Choose by execution
pattern first, then by control requirements.

Multi-agent orchestration patterns (``PlanExecutePattern``,
``ProposeCriticPattern``, ``TreeSearchPattern``, ``RalphLoopPattern``, ``RouterDelegatePattern``) live in the patterns module and are
implemented with the same public workflow step primitives available to users.

``SeededRandomBaselineAgent`` is a lightweight benchmarking/control-condition
participant for packaged-problem studies. It follows the same workflow-backed
runtime shape as the other public agents; supply packaged-problem objects
through the run-time ``dependencies`` mapping.

``PromptWorkflowAgent`` wraps a prompt-mode ``Workflow`` for packaged-problem
studies. Use it when the workflow is the real participant implementation but
the experiment loop should own problem-selection and binding timing, run IDs,
and condition selection. ``design-research-problems`` remains the owner of the
resolution and evaluation implementation.

Overview
--------

- ``DirectLLMCall``
- ``MultiStepAgent`` (``mode="direct" | "json" | "code"``)
- ``SeededRandomBaselineAgent``
- ``PromptWorkflowAgent``

Decision table
--------------

.. list-table::
   :header-rows: 1

   * - Use case
     - Recommended pattern
     - Why
   * - Plain text generation without tools
     - ``DirectLLMCall``
     - Lowest orchestration overhead
   * - Iterative direct (no external tools)
     - ``MultiStepAgent(mode="direct")``
     - Internal CONTINUE/STOP controller steps
   * - Iterative structured tool loops
     - ``MultiStepAgent(mode="json")``
     - ReAct-style multi-step loop with JSON ``tool_name``/``tool_input`` actions
   * - Iterative code-action loops
     - ``MultiStepAgent(mode="code")``
     - ReAct-style loop with code actions
   * - Packaged-problem control condition
     - ``SeededRandomBaselineAgent``
     - Thin seeded baseline over public problem contracts
   * - Packaged-problem prompt-mode workflow participant
     - ``PromptWorkflowAgent``
     - Reuses a public ``Workflow`` while letting experiments own run metadata

Background references
---------------------

- `ReAct <https://arxiv.org/abs/2210.03629>`_
- `Toolformer <https://arxiv.org/abs/2302.04761>`_
- `MRKL Systems <https://arxiv.org/abs/2205.00445>`_

These references are for conceptual grounding; behavior is defined by the
contracts and implementation in this repository.

Examples
--------

- :doc:`/examples/agents/index`
- ``examples/agents/README.md``

Pages
-----

- :doc:`multi_step_patterns`
- :doc:`background_reading`

.. toctree::
   :maxdepth: 2
   :hidden:

   multi_step_patterns
   background_reading
