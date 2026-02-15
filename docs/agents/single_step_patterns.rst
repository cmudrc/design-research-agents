Single-Step Patterns
====================

Single-step agents execute one model-guided action path per run.

Patterns
--------

``SingleStepDirectLLMAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- One direct completion with no tool calls.
- Best for baselines and text transforms.

``SingleStepRouterAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^

- Model selects exactly one route from runtime tool routes.
- Strong fit for strict one-shot route selection.

``SingleStepJsonToolCallingAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Model emits JSON tool call payload validated against runtime tools.
- Best for structured one-shot tool invocation.

``SingleStepCodeToolCallingAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Model writes one sandboxed Python action program.
- Best when a single step still requires multiple tool calls.

Examples
--------

- ``examples/agents/basic/single_step_direct_llm_agent.py``
- ``examples/agents/basic/single_step_router_agent.py``
- ``examples/agents/basic/single_step_json_tool_calling_agent.py``
- ``examples/agents/basic/single_step_code_tool_calling_agent.py``
- ``examples/agents/streaming/single_step_direct_llm_agent_stream.py``
- ``examples/agents/streaming/single_step_router_agent_stream.py``
- ``examples/agents/streaming/single_step_json_tool_calling_agent_stream.py``
- ``examples/agents/streaming/single_step_code_tool_calling_agent_stream.py``
- ``examples/agents/basic/README.md``
- ``examples/agents/streaming/README.md``
