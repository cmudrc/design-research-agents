Concepts
========

What Is An Agent In This Library?
---------------------------------

An agent is an executable participant that converts context into outputs under
explicit runtime contracts. The key design goal is controlled behavior, not only
text generation.

Direct Calls vs Multi-Step Agents
---------------------------------

``DirectLLMCall`` is a one-shot entry point with minimal orchestration overhead.
``MultiStepAgent`` provides iterative control loops for planning, tool use, and
stateful refinement.

Tools and Tool Routing
----------------------

Tools are exposed through ``Toolbox`` with three source types:

- callable tools,
- script tools, and
- MCP-backed tools.

Tool routing can be explicit (named calls) or agent-mediated via tool schemas.
In either case, the runtime records structured tool results for downstream
inspection.

Workflows vs Patterns
---------------------

Workflows are low-level execution graphs built from model/tool/delegate/loop/
memory steps. Patterns are higher-order reusable configurations built on top of
workflow primitives.

Use workflows when you need custom control. Use patterns when your study aligns
with standard coordination forms such as plan/execute, debate, routing, or RAG.

Traces and Observability
------------------------

Every run can emit structured execution metadata and traces. These records enable
behavioral comparison across prompts, tools, model backends, and control
strategies.

Clients and Execution Environments
----------------------------------

The package supports hosted APIs and local backends. Client selection is an
experimental choice: latency, privacy, hardware dependence, and reproducibility
constraints all affect study design.

Research Emphasis
-----------------

The library separates execution machinery from research abstractions so behavior
can be compared, audited, and reused across studies. The target output is not
only a response, but evidence about how the response was produced.
