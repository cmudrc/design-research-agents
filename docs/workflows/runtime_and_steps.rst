Workflow Runtime and Steps
==========================

``WorkflowRuntime`` executes typed steps with deterministic orchestration.

Top-level step exports
----------------------

All workflow step primitives are exported from the top-level package:

.. code-block:: python

   from design_research_agents import (
       AgentStep,
       LogicStep,
       LoopStep,
       MemoryReadStep,
       MemoryWriteStep,
       ToolStep,
   )

Step types
----------

- ``LogicStep``: deterministic local handlers.
  Key fields: ``step_id``, ``handler``, ``dependencies``, ``route_map``.
  Use it for pure orchestration logic, branching, and deterministic transforms.
- ``ToolStep``: tool runtime invocations.
  Key fields: ``step_id``, ``tool_name``, ``input_builder``, ``dependencies``.
  Use it for explicit tool boundary calls through ``Toolbox``.
- ``AgentStep``: delegated agent/workflow execution.
  Key fields: ``step_id``, ``agent_name``, ``prompt_builder``, ``dependencies``.
  Use it when nested agents/patterns should own their own prompting or tool usage.
- ``LoopStep``: iterative nested workflow body with loop state callbacks.
  Key fields: ``step_id``, ``steps``, ``max_iterations``, ``initial_state``,
  ``continue_predicate``, ``state_reducer``, ``execution_mode``, ``failure_policy``.
  Use it when iterative orchestration is first-class and state must evolve per iteration.
- ``MemoryReadStep``: retrieval step against a configured memory store.
  Key fields: ``step_id``, ``query_builder``, ``namespace``, ``top_k``, ``min_score``.
  Use it to inject retrieved context into downstream agent/tool steps.
- ``MemoryWriteStep``: persistence step for normalized memory records.
  Key fields: ``step_id``, ``records_builder``, ``namespace``.
  Use it to store intermediate/final artifacts for later retrieval.

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

Input mode contracts
--------------------

- ``input_mode='prompt'`` accepts non-empty string input only.
  It rejects non-string values and blank/whitespace-only prompts.
  Prompt input is provided to step context under ``prompt``.
- ``input_mode='schema'`` accepts mapping input only.
  It optionally validates payloads against ``input_schema``.
  Schema input is provided to step context under ``inputs``.
- Both modes support constructor-level run defaults and per-run overrides.
  They return ``AgentResult`` with serialized ``WorkflowResult`` metadata.

Examples
--------

- ``examples/workflow/workflow_runtime.py``
- ``examples/workflow/workflow_runtime_loop_step.py``
- ``examples/workflow/workflow_schema_mode.py``
- ``examples/workflow/workflow_prompt_mode.py``
