Single-Step Patterns
====================

Single-step agents execute one model-guided action path per run.

Patterns
--------

``SingleStepDirectLLMAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- One direct completion with no tool calls.
- Best for baselines and text transforms.
- Constructor kwargs now use ``system_prompt`` (replacing the old
  ``default_system_prompt`` name).

``SingleStepToolRouterAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Model selects exactly one route from runtime tool routes.
- Strong fit for strict one-shot route selection.
- Supports constructor-time overrides for ``system_prompt``,
  ``user_prompt_template``, ``alternatives_prompt_target``, and
  ``allowed_routes``.

``SingleStepJsonToolCallingAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Model emits JSON tool call payload validated against runtime tools.
- Best for structured one-shot tool invocation.
- Supports constructor-time overrides for ``system_prompt``,
  ``user_prompt_template``, ``alternatives_prompt_target``, and
  ``allowed_tools``.

``SingleStepCodeToolCallingAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Model writes one sandboxed Python action program.
- Best when a single step still requires multiple tool calls.
- Supports constructor-time overrides for ``system_prompt``,
  ``user_prompt_template``, and ``alternatives_prompt_target`` in addition to
  execution guardrail kwargs.

Examples
--------

- ``examples/agents/basic/single_step_direct_llm_agent.py``
- ``examples/agents/basic/single_step_tool_router_agent.py``
- ``examples/agents/basic/single_step_json_tool_calling_agent.py``
- ``examples/agents/basic/single_step_json_callable_tool_agent.py``
- ``examples/agents/basic/single_step_code_tool_calling_agent.py``
- ``examples/agents/streaming/single_step_direct_llm_agent_stream.py``
- ``examples/agents/streaming/single_step_tool_router_agent_stream.py``
- ``examples/agents/streaming/single_step_json_tool_calling_agent_stream.py``
- ``examples/agents/streaming/single_step_code_tool_calling_agent_stream.py``
- ``examples/agents/basic/README.md``
- ``examples/agents/streaming/README.md``
