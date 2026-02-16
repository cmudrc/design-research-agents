Workflow Runtime and Steps
==========================

``WorkflowRuntime`` executes typed steps with deterministic orchestration.

Top-level step exports
----------------------

All workflow step primitives are exported from the top-level package:

.. code-block:: python

   from design_research_agents import AgentStep, LogicStep, LoopStep, ToolStep

Step types
----------

- ``LogicStep``: deterministic local handlers
- ``ToolStep``: tool runtime invocations
- ``AgentStep``: delegated agent/workflow execution
- ``LoopStep``: iterative nested workflow body with loop state callbacks

Loop primitive
--------------

- ``LoopStep`` executes a fixed nested step sequence for up to ``max_iterations``.
- ``continue_predicate`` can stop early based on iteration index and loop state.
- ``state_reducer`` updates loop state from each iteration ``WorkflowResult``.
- Loop step outputs include explicit termination reason and serialized
  per-iteration results.

Execution semantics
-------------------

- ``execution_mode=\"sequential\"`` for strict order
- ``execution_mode=\"dag\"`` for dependency-aware scheduling
- Supports dependency injection through ``dependency_results``
- Supports route-based branch skipping via ``LogicStep.route_map``
- Supports configurable failure policies

Reusable facade
---------------

``Workflow`` is the high-level constructor-first facade for user-defined graphs:

- ``Workflow(input_mode='prompt')`` for string prompt input.
- ``Workflow(input_mode='schema')`` for mapping input with optional schema validation.
- Supports optional ``agents`` registration for ``AgentStep`` delegates.

Examples
--------

- ``examples/workflow/workflow_runtime.py``
- ``examples/workflow/workflow_runtime_loop_step.py``
- ``examples/workflow/pure_tool_workflow.py``
- ``examples/workflow/mixed_agent_workflow.py``
