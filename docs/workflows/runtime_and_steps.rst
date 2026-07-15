Workflow Runtime and Steps
==========================

``Workflow`` is the public orchestration facade for typed step execution.
``WorkflowRuntime`` remains the internal engine that powers ``Workflow`` and
prebuilt pattern implementations.

``design_research_agents.workflow`` is the public module for constructing new
workflow implementations. Prebuilt implementations live in
``design_research_agents.patterns``.

Workflow-module step exports
----------------------------

All workflow step primitives are available from the workflow module:

.. code-block:: python

   from design_research_agents.workflow import (
       DelegateStep,
       DelegateBatchStep,
       LogicStep,
       LoopStep,
       MemoryReadStep,
       MemoryWriteStep,
       ModelStep,
       ToolStep,
       Workflow,
       WorkflowContext,
   )

Step types
----------

- ``LogicStep``: deterministic local handlers.
  Key fields: ``step_id``, ``handler``, ``dependencies``, ``route_map``.
  Use it for pure orchestration logic, branching, and deterministic transforms.
- ``ToolStep``: tool runtime invocations.
  Key fields: ``step_id``, ``tool_name``, ``input_builder``, ``dependencies``.
  Use it for explicit tool boundary calls through ``Toolbox``.
- ``DelegateStep``: delegated agent/workflow execution.
  Key fields: ``step_id``, ``delegate``, ``prompt_builder``, ``dependencies``.
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
- ``state_reducer`` updates loop state from each iteration ``ExecutionResult``.
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

- ``Workflow(input_schema=None)`` (default) for string prompt input.
- ``Workflow(input_schema={...})`` for mapping input with schema validation.
- ``Workflow(output_schema={...})`` to validate canonical ``output.final_output``.
- ``DelegateStep`` delegates are provided directly on each step object.

Input mode contracts
--------------------

- ``input_schema=None`` accepts non-empty string input only.
  It rejects non-string values and blank/whitespace-only prompts.
  Prompt input is provided to step context under ``prompt``.
- ``input_schema={...}`` accepts mapping input only.
  It validates payloads against ``input_schema``.
  Schema input is provided to step context under ``inputs``.
- ``output_schema={...}`` validates ``output.final_output`` when workflow execution succeeds.
  Use ``scalar(...)``, ``typed_dict(...)``, and ``list_of(...)`` from
  ``design_research_agents.workflow`` for concise schema composition.
- Both modes support constructor-level run defaults and per-run overrides.
  They return ``ExecutionResult`` with consistent workflow metadata.
- Prebuilt implementations in ``design_research_agents.patterns`` are authored
  with these same primitives.

Callback context
----------------

Step builders and handlers receive ``WorkflowContext``. It remains a
read-only ``Mapping[str, object]``, so existing dictionary-style callbacks do
not need to change. New callbacks can use typed accessors to avoid unpacking
nested dependency dictionaries:

.. code-block:: python

   def summarize(context: WorkflowContext) -> dict[str, object]:
       source = context.dependency_output("source")
       return {
           "title": context.input_value("title", "Untitled"),
           "count": source.get("count", 0),
       }

``context.inputs`` contains schema-mode inputs, ``dependency_result(step_id)``
returns the typed ``WorkflowStepResult``, and ``request_id`` / ``step_id``
surface the current runtime metadata. The legacy
``context["dependency_results"]`` serialization remains available.

Examples
--------

- ``examples/workflow/workflow_runtime.py``
- ``examples/workflow/workflow_runtime_loop_step.py``
- ``examples/workflow/workflow_schema_mode.py``
- ``examples/workflow/workflow_prompt_mode.py``
