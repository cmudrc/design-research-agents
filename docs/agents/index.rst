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
   * - Iterative tool routing loop
     - ``MultiStepAgent(mode="json")`` with arg-less tools
     - ReAct-style TOOL_CALL/STOP controller loop (auto-special-case)
   * - Iterative structured tool loops
     - ``MultiStepAgent(mode="json")``
     - ReAct-style multi-step loop with JSON actions
   * - Iterative code-action loops
     - ``MultiStepAgent(mode="code")``
     - ReAct-style loop with code actions

Examples
--------

- ``examples/agents/basic/README.md``

Pages
-----

- :doc:`multi_step_patterns`
- :doc:`background_reading`

.. toctree::
   :maxdepth: 2
   :hidden:

   multi_step_patterns
   background_reading
