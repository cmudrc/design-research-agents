Agent Types
===========

This framework exposes six core concrete agent implementations:

- ``SingleStepDirectLLMAgent``
- ``SingleStepRouterAgent``
- ``SingleStepJsonToolCallingAgent``
- ``SingleStepCodeToolCallingAgent``
- ``MultiStepJsonToolCallingAgent``
- ``MultiStepCodeToolCallingAgent``

All core agent types implement ``dra.contracts.agent.Agent`` and return
``AgentResult``.

Facade access:

.. code-block:: python

   import design_research_agents as dra

   direct = dra.agents.SingleStepDirectLLMAgent(...)
   router = dra.agents.SingleStepRouterAgent(...)
   multi_json = dra.agents.MultiStepJsonToolCallingAgent(...)

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
   * - ``SingleStepDirectLLMAgent``
     - One direct chat completion with no tools.
     - Baselines, text transforms, and simple prompt-response tasks.
     - Prompt string.
   * - ``SingleStepRouterAgent``
     - Model-driven route selection + one tool call.
     - One-shot routing when strict model-validated selection is required.
     - Prompt string (routes come from runtime tools).
   * - ``SingleStepJsonToolCallingAgent``
     - Model emits one JSON tool call, then runtime executes it.
     - LLM-driven single-step tool selection with structured arguments.
     - Prompt string (tool choices come from runtime tools).
   * - ``SingleStepCodeToolCallingAgent``
     - Model writes one sandboxed Python action program.
     - One-step plans that still need multiple tool calls.
     - Prompt string.
   * - ``MultiStepJsonToolCallingAgent``
     - ReAct-style loop over ``SingleStepJsonToolCallingAgent`` steps.
     - Iterative tasks solved by repeated JSON tool-call steps.
     - Prompt string.
   * - ``MultiStepCodeToolCallingAgent``
     - ReAct-style loop over ``SingleStepCodeToolCallingAgent`` steps.
     - Iterative tasks that require tool chains via generated code.
     - Prompt string.

SingleStepDirectLLMAgent
------------------------

- Source: ``src/design_research_agents/agent/implementations/single_step_direct_llm_agent.py``
- Example: ``examples/agents/basic/single_step_direct_llm_agent.py``
- Streaming example: ``examples/agents/streaming/single_step_direct_llm_agent_stream.py``
- Calls ``LLMClient`` directly and returns model output with no tool invocations.
- Uses the prompt string as a single user message (optional default system prompt can be set at init).
- Resolves model strictly from ``llm_client.default_model()``.

SingleStepRouterAgent
---------------------

- Source: ``src/design_research_agents/agent/implementations/single_step_router_agent.py``
- Example: ``examples/agents/basic/single_step_router_agent.py``
- Streaming example: ``examples/agents/streaming/single_step_router_agent_stream.py``
- Chooses exactly one runtime tool route from model-generated structured output.
- ``SingleStepRouterAgent`` is a tool-routing agent (not an agent-delegation orchestrator).
- Compiles route choices directly from ``ToolRuntime.list_tools()``.
- Uses a built-in default route-selection schema generated from runtime routes.
- Fails the run when model routing output is invalid (no heuristic route fallback).
- Stores route traces in ``AgentResult.metadata["routing"]``.

SingleStepJsonToolCallingAgent
------------------------------

- Source: ``src/design_research_agents/agent/implementations/single_step_json_tool_calling_agent.py``
- Example: ``examples/agents/basic/single_step_json_tool_calling_agent.py``
- Streaming example: ``examples/agents/streaming/single_step_json_tool_calling_agent_stream.py``
- Requests JSON-only tool-call output with a built-in schema compiled from runtime tools.
- Validates model-selected tool names against runtime-registered tools.
- Falls back to heuristic tool selection when model output is invalid.

SingleStepCodeToolCallingAgent
------------------------------

- Source: ``src/design_research_agents/agent/implementations/single_step_code_tool_calling_agent.py``
- Example: ``examples/agents/basic/single_step_code_tool_calling_agent.py``
- Streaming example: ``examples/agents/streaming/single_step_code_tool_calling_agent_stream.py``
- Uses init-compiled runtime tools and executes generated code in a strict sandbox.
- Enforces limits such as ``max_tool_calls`` and ``execution_timeout_seconds``.
- Can optionally validate generated tool arguments against tool input schemas.

MultiStepJsonToolCallingAgent
-----------------------------

- Source: ``src/design_research_agents/agent/implementations/multi_step_json_tool_calling_agent.py``
- Example: ``examples/agents/basic/multi_step_json_tool_calling_agent.py``
- Streaming example: ``examples/agents/streaming/multi_step_json_tool_calling_agent_stream.py``
- Repeats action-observation cycles with memory until continuation stops or limits are reached.
- Uses a built-in default continuation-decision schema for each loop iteration.
- Delegates each step to ``SingleStepJsonToolCallingAgent``.
- Returns step traces and continuation metadata for observability/debugging.

MultiStepCodeToolCallingAgent
-----------------------------

- Source: ``src/design_research_agents/agent/implementations/multi_step_code_tool_calling_agent.py``
- Example: ``examples/agents/basic/multi_step_code_tool_calling_agent.py``
- Streaming example: ``examples/agents/streaming/multi_step_code_tool_calling_agent_stream.py``
- Repeats action-observation cycles with memory until continuation stops or limits are reached.
- Uses a built-in default continuation-decision schema for each loop iteration.
- Uses model-based continuation decisions with deterministic fallback heuristics.
- Can forward init-time ``default_tools_per_step`` into each step agent.
- Returns step traces and continuation metadata for observability/debugging.

AgentRuntime
------------

``AgentRuntime`` is a runtime facade, not one of the six core implementations.

- Source: ``src/design_research_agents/agent/runtime.py``
- Examples:
  - ``examples/orchestrator/plan_execute.py``
  - ``examples/orchestrator/propose_critic.py``
  - ``examples/orchestrator/agent_routing.py``
- Interaction pattern mirrors agent examples: construct first, then call
  ``.run(prompt=...)`` with an explicit prompt.
- Provides one runtime that can execute:
  - ``mode="react"`` (delegates directly to ``MultiStepCodeToolCallingAgent``),
  - ``mode="plan_execute"`` (planner JSON + step execution),
  - ``mode="propose_critic"`` (iterative propose/critic loop),
  - ``mode="agent_routing"`` (tool-routing selection + delegated agent execution).
- Tracks soft budget metadata (latency/cost observations) across mode loops.

Workflow Runtime
----------------

- Source:
  - ``src/design_research_agents/orchestrator/implementations/workflow_runtime.py``
- Reusable orchestration chunks:
  - ``src/design_research_agents/orchestrator/implementations/agent_routing.py``
  - ``src/design_research_agents/orchestrator/implementations/plan_execute.py``
  - ``src/design_research_agents/orchestrator/implementations/propose_critic.py``
  - ``src/design_research_agents/orchestrator/implementations/pure_tool_workflow.py``
  - ``src/design_research_agents/orchestrator/implementations/mixed_agent_workflow.py``
- Examples:
  - ``examples/orchestrator/workflow_runtime.py``
  - ``examples/orchestrator/plan_execute.py``
  - ``examples/orchestrator/propose_critic.py``
  - ``examples/orchestrator/agent_routing.py``
  - ``examples/orchestrator/pure_tool_workflow.py``
  - ``examples/orchestrator/mixed_agent_workflow.py``
- ``WorkflowRuntime`` executes typed workflow steps:
  - ``LogicStep`` for deterministic local handlers,
  - ``ToolStep`` for runtime tool invocations,
  - ``AgentStep`` for registered agent-or-orchestrator delegation.
- Reusable orchestration chunks align to the
  ``dra.contracts.orchestrator.WorkflowOrchestrator`` protocol.
- Supports deterministic ``execution_mode="sequential"`` and
  ``execution_mode="dag"`` scheduling, per-step dependency injection through
  ``dependency_results``, route-based branch skipping via ``LogicStep.route_map``,
  and configurable failure-policy behavior.
