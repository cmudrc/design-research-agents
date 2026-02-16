Multi-Step Patterns
===================

Multi-step agents execute iterative action-observation loops until a
continuation decision stops.

Patterns
--------

``MultiStepJsonToolCallingAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ReAct-style loop over JSON tool-call actions.
- Strong fit for structured iterative decomposition.
- Constructor kwargs expose continuation/step prompt overrides plus
  ``continuation_memory_tail_items`` and ``step_memory_tail_items`` controls.

``MultiStepCodeToolCallingAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ReAct-style loop over generated code actions.
- Strong fit for iterative tool chains that need richer control flow.
- Constructor kwargs expose continuation/step prompt overrides, alternatives
  placement, and per-step memory-tail controls.

Continuation and limits
-----------------------

Both multi-step agents rely on continuation decisions and runtime limits
(step/tool-call/time constraints) to bound execution.

Examples
--------

- ``examples/agents/basic/multi_step_json_tool_calling_agent.py``
- ``examples/agents/basic/multi_step_code_tool_calling_agent.py``
- ``examples/agents/streaming/multi_step_json_tool_calling_agent_stream.py``
- ``examples/agents/streaming/multi_step_code_tool_calling_agent_stream.py``
- ``examples/agents/basic/README.md``
- ``examples/agents/streaming/README.md``
