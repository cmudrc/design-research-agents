Workflow Runtime and Steps
==========================

``WorkflowRuntime`` executes typed steps with deterministic orchestration.

Step types
----------

- ``LogicStep``: deterministic local handlers
- ``ToolStep``: tool runtime invocations
- ``AgentStep``: delegated agent/workflow execution

Execution semantics
-------------------

- ``execution_mode=\"sequential\"`` for strict order
- ``execution_mode=\"dag\"`` for dependency-aware scheduling
- Supports dependency injection through ``dependency_results``
- Supports route-based branch skipping via ``LogicStep.route_map``
- Supports configurable failure policies

Examples
--------

- ``examples/workflow/workflow_runtime.py``
- ``examples/workflow/pure_tool_workflow.py``
- ``examples/workflow/mixed_agent_workflow.py``
