Agents
======

The framework provides six core concrete agent implementations. Choose by
execution pattern first, then by control requirements.

Overview
--------

- ``SingleStepDirectLLMAgent``
- ``SingleStepRouterAgent``
- ``SingleStepJsonToolCallingAgent``
- ``SingleStepCodeToolCallingAgent``
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
     - ``SingleStepDirectLLMAgent``
     - Lowest orchestration overhead
   * - One-shot model-driven tool route selection
     - ``SingleStepRouterAgent``
     - Strict route schema with one route execution
   * - One-shot structured tool call
     - ``SingleStepJsonToolCallingAgent``
     - JSON-validated tool/args output path
   * - One-shot multi-call tool choreography
     - ``SingleStepCodeToolCallingAgent``
     - Generated code can call multiple tools in one step
   * - Iterative structured tool loops
     - ``MultiStepJsonToolCallingAgent``
     - ReAct-style multi-step loop with JSON actions
   * - Iterative code-action loops
     - ``MultiStepCodeToolCallingAgent``
     - ReAct-style loop with code actions

Examples
--------

- ``examples/agents/basic/README.md``
- ``examples/agents/streaming/README.md``

Pages
-----

- :doc:`single_step_patterns`
- :doc:`multi_step_patterns`
- :doc:`background_reading`

.. toctree::
   :maxdepth: 2
   :hidden:

   single_step_patterns
   multi_step_patterns
   background_reading
