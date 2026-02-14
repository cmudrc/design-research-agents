Agent Types
===========

This framework exposes four concrete agent implementations:

- ``RouterAgent``
- ``ToolCallingAgent``
- ``SingleStepCodeAgent``
- ``MultiStepAgent``

All agent types implement ``design_research_agents.contracts.agent.Agent`` and
return ``AgentResult``.

Shared Contract
---------------

.. code-block:: python

   result = agent.run(input=input_payload, context=context_payload)
   for event in agent.run_stream(input=input_payload, context=context_payload):
       ...

Comparison
----------

.. list-table::
   :header-rows: 1

   * - Agent
     - Execution pattern
     - Best fit
     - Key input fields
   * - ``RouterAgent``
     - Heuristic route selection + one model call + one tool call.
     - Fast, deterministic one-shot tool routing.
     - ``prompt`` with ``alternatives`` (or legacy ``tool_name`` fallback).
   * - ``ToolCallingAgent``
     - Model emits one JSON tool call, then runtime executes it.
     - LLM-driven tool selection with structured arguments.
     - ``prompt`` with ``tools``/``alternatives`` (or runtime tool list fallback).
   * - ``SingleStepCodeAgent``
     - Model writes one sandboxed Python action program.
     - One-step plans that still need multiple tool calls.
     - ``prompt`` with explicit ``tools`` list (required).
   * - ``MultiStepAgent``
     - ReAct-style loop over ``SingleStepCodeAgent`` steps.
     - Iterative tasks with action-observation memory.
     - ``prompt`` with ``tools`` plus optional loop controls (for example ``max_steps``).

RouterAgent
-----------

- Source: ``src/design_research_agents/agent/router_agent.py``
- Example: ``examples/router_agent.py``
- Chooses exactly one tool alternative using token overlap plus math/text heuristics.
- Supports a fallback default tool when alternatives are omitted.
- Stores route traces in ``AgentResult.metadata["routing"]``.

ToolCallingAgent
----------------

- Source: ``src/design_research_agents/agent/tool_calling_agent.py``
- Example: ``examples/tool_calling_agent.py``
- Requests JSON-only tool-call output from the model with a constrained response schema.
- Validates model-selected tool names against the available set.
- Falls back to heuristic tool selection when model output is invalid.

SingleStepCodeAgent
-------------------

- Source: ``src/design_research_agents/agent/single_step_code_agent.py``
- Example: ``examples/single_step_code_agent.py``
- Requires explicit allowed tools and executes generated code in a strict sandbox.
- Enforces limits such as ``max_tool_calls`` and ``execution_timeout_seconds``.
- Can optionally validate generated tool arguments against tool input schemas.

MultiStepAgent
--------------

- Source: ``src/design_research_agents/agent/multi_step_agent.py``
- Example: ``examples/multi_step_agent.py``
- Repeats action-observation cycles with memory until continuation stops or limits are reached.
- Uses model-based continuation decisions with deterministic fallback heuristics.
- Returns step traces and continuation metadata for observability/debugging.
