Agent Types
===========

This framework exposes six concrete agent implementations:

- ``DirectLLMAgent``
- ``RouterAgent``
- ``ToolCallingAgent``
- ``SingleStepCodeAgent``
- ``MultiStepAgent``
- ``AgentRuntime``

All agent types implement ``design_research_agents.contracts.agent.Agent`` and
return ``AgentResult``.

Shared Contract
---------------

.. code-block:: python

   result = agent.run(prompt="Summarize this changelog.", request_id=request_id, dependencies=dependencies)
   for event in agent.run_stream(prompt="Stream a short greeting."):
       ...

``prompt`` is a plain prompt string.

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
     - Prompt string.
   * - ``RouterAgent``
     - Model-driven route selection + one tool call.
     - One-shot routing when strict model-validated selection is required.
     - Prompt string (routes come from runtime tools).
   * - ``ToolCallingAgent``
     - Model emits one JSON tool call, then runtime executes it.
     - LLM-driven tool selection with structured arguments.
     - Prompt string (tool choices come from runtime tools).
   * - ``SingleStepCodeAgent``
     - Model writes one sandboxed Python action program.
     - One-step plans that still need multiple tool calls.
     - Prompt string.
   * - ``MultiStepAgent``
     - ReAct-style loop over ``SingleStepCodeAgent`` steps.
     - Iterative tasks with action-observation memory.
     - Prompt string.
   * - ``AgentRuntime``
     - Unified mode-based runtime (react, plan_execute, propose_critic, triage).
     - One entrypoint when you want to swap execution patterns.
     - Prompt string plus constructor-time runtime controls.

DirectLLMAgent
--------------

- Source: ``src/design_research_agents/agent/direct_llm_agent.py``
- Example: ``examples/agents/basic/direct_llm_agent.py``
- Streaming example: ``examples/agents/streaming/direct_llm_agent_stream.py``
- Calls ``LLMClient`` directly and returns model output with no tool invocations.
- Uses the prompt string as a single user message (optional default system prompt can be set at init).
- Resolves model names with the same precedence as other agents.

RouterAgent
-----------

- Source: ``src/design_research_agents/agent/router_agent.py``
- Example: ``examples/agents/basic/router_agent.py``
- Streaming example: ``examples/agents/streaming/router_agent_stream.py``
- Chooses exactly one runtime tool route from model-generated structured output.
- Compiles route choices directly from ``ToolRuntime.list_tools()``.
- Uses a built-in default route-selection schema generated from runtime routes.
- Fails the run when model routing output is invalid (no heuristic route fallback).
- Stores route traces in ``AgentResult.metadata["routing"]``.

ToolCallingAgent
----------------

- Source: ``src/design_research_agents/agent/tool_calling_agent.py``
- Example: ``examples/agents/basic/tool_calling_agent.py``
- Streaming example: ``examples/agents/streaming/tool_calling_agent_stream.py``
- Requests JSON-only tool-call output with a built-in schema compiled from runtime tools.
- Validates model-selected tool names against runtime-registered tools.
- Falls back to heuristic tool selection when model output is invalid.

SingleStepCodeAgent
-------------------

- Source: ``src/design_research_agents/agent/single_step_code_agent.py``
- Example: ``examples/agents/basic/single_step_code_agent.py``
- Streaming example: ``examples/agents/streaming/single_step_code_agent_stream.py``
- Uses init-compiled runtime tools and executes generated code in a strict sandbox.
- Enforces limits such as ``max_tool_calls`` and ``execution_timeout_seconds``.
- Can optionally validate generated tool arguments against tool input schemas.

MultiStepAgent
--------------

- Source: ``src/design_research_agents/agent/multi_step_agent.py``
- Example: ``examples/agents/basic/multi_step_agent.py``
- Streaming example: ``examples/agents/streaming/multi_step_agent_stream.py``
- Repeats action-observation cycles with memory until continuation stops or limits are reached.
- Uses a built-in default continuation-decision schema for each loop iteration.
- Uses model-based continuation decisions with deterministic fallback heuristics.
- Can forward init-time ``default_tools_per_step`` into each step agent.
- Returns step traces and continuation metadata for observability/debugging.

AgentRuntime
------------

- Source: ``src/design_research_agents/agent/runtime.py``
- Examples:
  - ``examples/runtime/plan_execute.py``
  - ``examples/runtime/propose_critic.py``
  - ``examples/runtime/triage.py``
- Provides one runtime that can execute:
  - ``mode=\"react\"`` (delegates directly to ``MultiStepAgent``),
  - ``mode=\"plan_execute\"`` (planner JSON + step execution),
  - ``mode=\"propose_critic\"`` (iterative propose/critic loop),
  - ``mode=\"triage\"`` (router selection + delegated agent execution).
- Tracks soft budget metadata (latency/cost observations) across mode loops.

Workflow Runtime
----------------

- Source:
  - ``src/design_research_agents/orchestrator/runtime.py``
- Examples:
  - ``examples/orchestrator/pure_tool_workflow.py``
  - ``examples/orchestrator/mixed_agent_workflow.py``
- ``WorkflowRuntime`` executes typed workflow steps:
  - ``LogicStep`` for deterministic local handlers,
  - ``ToolStep`` for runtime tool invocations,
  - ``AgentStep`` for registered-agent delegation.
- Supports deterministic ``execution_mode=\"sequential\"`` and
  ``execution_mode=\"dag\"`` scheduling, per-step dependency injection through
  ``dependency_results``, route-based branch skipping via ``LogicStep.route_map``,
  and configurable failure-policy behavior.
