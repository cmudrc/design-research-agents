Agent Types
===========

This framework exposes five concrete agent implementations:

- ``DirectLLMAgent``
- ``RouterAgent``
- ``ToolCallingAgent``
- ``SingleStepCodeAgent``
- ``MultiStepAgent``

All agent types implement ``design_research_agents.contracts.agent.Agent`` and
return ``AgentResult``.

Shared Contract
---------------

.. code-block:: python

   result = agent.run(input="Summarize this changelog.", request_id=request_id, dependencies=dependencies)
   result = agent.run(input={"prompt": "Summarize this changelog.", "model": "my-model"})
   for event in agent.run_stream(input="Stream a short greeting."):
       ...

``input`` accepts either a structured mapping payload or plain string shorthand
for ``{"prompt": <input>}``.

Comparison
----------

.. list-table::
   :header-rows: 1

   * - Agent
     - Execution pattern
     - Best fit
     - Key input fields
   * - ``DirectLLMAgent``
     - One direct chat completion with no tools.
     - Baselines, text transforms, and simple prompt-response tasks.
     - ``prompt``/``text`` or explicit ``messages`` with optional generation params and optional ``alternatives_prompt_target``.
   * - ``RouterAgent``
     - Model-driven route selection + one tool call.
     - One-shot routing when strict model-validated selection is required.
     - ``prompt`` (routes come from runtime tools), optional ``alternatives_prompt_target``.
   * - ``ToolCallingAgent``
     - Model emits one JSON tool call, then runtime executes it.
     - LLM-driven tool selection with structured arguments.
     - ``prompt`` (tool choices come from runtime tools), optional ``alternatives_prompt_target``.
   * - ``SingleStepCodeAgent``
     - Model writes one sandboxed Python action program.
     - One-step plans that still need multiple tool calls.
     - ``prompt`` plus optional execution controls and optional ``alternatives_prompt_target``.
   * - ``MultiStepAgent``
     - ReAct-style loop over ``SingleStepCodeAgent`` steps.
     - Iterative tasks with action-observation memory.
     - ``prompt`` plus optional loop controls (for example ``max_steps``) and optional ``alternatives_prompt_target``.

DirectLLMAgent
--------------

- Source: ``src/design_research_agents/agent/direct_llm_agent.py``
- Example: ``examples/basic/direct_llm_agent.py``
- Streaming example: ``examples/streaming/direct_llm_agent_stream.py``
- Calls ``LLMClient`` directly and returns model output with no tool invocations.
- Supports either explicit ``messages`` input or ``prompt``/``text`` fallback.
- Can inject optional ``input["alternatives"]`` into the system or user prompt using ``alternatives_prompt_target``.
- Resolves model names with the same precedence as other agents.

RouterAgent
-----------

- Source: ``src/design_research_agents/agent/router_agent.py``
- Example: ``examples/basic/router_agent.py``
- Streaming example: ``examples/streaming/router_agent_stream.py``
- Chooses exactly one runtime tool route from model-generated structured output.
- Compiles route choices directly from ``ToolRuntime.list_tools()``.
- Uses a built-in default route-selection schema generated from runtime routes.
- Supports routing runtime alternatives context into either system or user prompt via ``alternatives_prompt_target``.
- Fails the run when model routing output is invalid (no heuristic route fallback).
- Stores route traces in ``AgentResult.metadata["routing"]``.

ToolCallingAgent
----------------

- Source: ``src/design_research_agents/agent/tool_calling_agent.py``
- Example: ``examples/basic/tool_calling_agent.py``
- Streaming example: ``examples/streaming/tool_calling_agent_stream.py``
- Requests JSON-only tool-call output with a built-in schema compiled from runtime tools.
- Validates model-selected tool names against runtime-registered tools.
- Supports routing runtime tool-choice context into either system or user prompt via ``alternatives_prompt_target``.
- Falls back to heuristic tool selection when model output is invalid.

SingleStepCodeAgent
-------------------

- Source: ``src/design_research_agents/agent/single_step_code_agent.py``
- Example: ``examples/basic/single_step_code_agent.py``
- Streaming example: ``examples/streaming/single_step_code_agent_stream.py``
- Uses init-compiled runtime tools and executes generated code in a strict sandbox.
- Enforces limits such as ``max_tool_calls`` and ``execution_timeout_seconds``.
- Supports routing allowed-tool context into either system or user prompt via ``alternatives_prompt_target``.
- Can optionally validate generated tool arguments against tool input schemas.

MultiStepAgent
--------------

- Source: ``src/design_research_agents/agent/multi_step_agent.py``
- Example: ``examples/basic/multi_step_agent.py``
- Streaming example: ``examples/streaming/multi_step_agent_stream.py``
- Repeats action-observation cycles with memory until continuation stops or limits are reached.
- Uses a built-in default continuation-decision schema for each loop iteration.
- Uses model-based continuation decisions with deterministic fallback heuristics.
- Supports routing step-tool alternatives context into either system or user prompt via ``alternatives_prompt_target``.
- Can forward init-time ``default_tools_per_step`` into each step agent.
- Returns step traces and continuation metadata for observability/debugging.
