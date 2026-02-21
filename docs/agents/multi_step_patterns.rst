Multi-Step Patterns
===================

Multi-step agents execute iterative action-observation loops until a
continuation decision stops.

Patterns
--------

``MultiStepAgent(mode="direct")``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Iterative direct-response loop with internal ``CONTINUE``/``STOP`` decisions.
- No external tool calls; each step refines or finalizes the answer.

``MultiStepAgent(mode="json")`` router special-case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ReAct-style loop where each step emits either ``TOOL_CALL`` or ``STOP``.
- ``TOOL_CALL`` selects route candidates via ``tool_names`` and executes the first valid route.

``MultiStepAgent(mode="json")``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ReAct-style loop over JSON tool-call actions.
- Strong fit for structured iterative decomposition.
- Constructor kwargs expose continuation/step prompt overrides plus
  ``continuation_memory_tail_items`` and ``step_memory_tail_items`` controls.

``MultiStepAgent(mode="code")``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ReAct-style loop over generated code actions.
- Strong fit for iterative tool chains that need richer control flow.
- Constructor kwargs expose continuation/step prompt overrides, alternatives
  placement, and per-step memory-tail controls.

Continuation and limits
-----------------------

All multi-step agents rely on continuation decisions and runtime limits
(step/tool-call constraints) to bound execution.

Examples
--------

- ``examples/agents/basic/multi_step_direct_llm_agent.py``
- ``examples/agents/basic/multi_step_json_tool_calling_agent.py``
- ``examples/agents/basic/multi_step_code_tool_calling_agent.py``
- ``examples/agents/basic/README.md``
