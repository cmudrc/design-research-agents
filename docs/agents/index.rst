Agents
======

The framework provides two core concrete agent entry points. Choose by
execution pattern first, then by control requirements.

Multi-agent orchestration patterns (``PlannerExecutorPattern``,
``ReflexionPattern``, ``RouterPattern``) live in the workflow module and are
implemented with the same public workflow step primitives available to users.

Overview
--------

- ``DirectLLMCall``
- ``MultiStepAgent`` (``mode="direct" | "json" | "code"``)

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
