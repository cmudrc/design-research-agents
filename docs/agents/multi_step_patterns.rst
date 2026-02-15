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

``MultiStepCodeToolCallingAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ReAct-style loop over generated code actions.
- Strong fit for iterative tool chains that need richer control flow.

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
