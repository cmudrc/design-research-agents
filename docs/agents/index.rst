Agents
======

The framework provides five core concrete agent implementations. Choose by
execution pattern first, then by control requirements.

Multi-agent orchestration patterns (``PlannerExecutorPattern``,
``ReflexionPattern``, ``RouterPattern``) live in the workflow module and are
implemented with the same public workflow step primitives available to users.

Overview
--------

- ``DirectLLMCall``
- ``MultiStepDirectLLMAgent``
- ``MultiStepToolRouterAgent``
- ``MultiStepJsonToolCallingAgent``
- ``MultiStepCodeToolCallingAgent``

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
     - ``MultiStepDirectLLMAgent``
     - Internal CONTINUE/STOP controller steps
   * - Iterative tool routing loop
     - ``MultiStepToolRouterAgent``
     - ReAct-style TOOL_CALL/STOP controller loop
   * - Iterative structured tool loops
     - ``MultiStepJsonToolCallingAgent``
     - ReAct-style multi-step loop with JSON actions
   * - Iterative code-action loops
     - ``MultiStepCodeToolCallingAgent``
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
