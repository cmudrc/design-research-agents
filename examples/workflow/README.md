## Workflow Runtime Examples

These entrypoints exercise orchestration-level flows, including reusable
workflow chunks in `src/design_research_agents/workflow/implementations/`.
Most chunks follow agent-like reuse semantics:
initialize once, then call `.run(...)` repeatedly with per-run input.

## What Each Example Demonstrates

- `workflow_runtime.py`
  - Direct `WorkflowRuntime` orchestration with composed steps.
- `plan_execute.py`
  - Planner + executor pattern using runtime tools.
- `propose_critic.py`
  - Propose/critique revision loop.
- `agent_routing.py`
  - Intent/agent routing with delegate execution.
- `pure_tool_workflow.py`
  - User-defined pure tool/logic step graph with `run(inputs=...)`.
- `mixed_agent_workflow.py`
  - User-defined mixed (logic + agent + tool) step graph with `run(prompt=...)`.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/workflow/workflow_runtime.py
PYTHONPATH=src python3 examples/workflow/plan_execute.py
PYTHONPATH=src python3 examples/workflow/propose_critic.py
PYTHONPATH=src python3 examples/workflow/agent_routing.py
PYTHONPATH=src python3 examples/workflow/pure_tool_workflow.py
PYTHONPATH=src python3 examples/workflow/mixed_agent_workflow.py
```

## Implementation Mapping

- `workflow_runtime.py` (`WorkflowRuntime`) -> `examples/workflow/workflow_runtime.py`
- `plan_execute.py` (`PlanExecuteWorkflow`) -> `examples/workflow/plan_execute.py`
- `propose_critic.py` (`ProposeAndCritiqueWorkflow`) -> `examples/workflow/propose_critic.py`
- `agent_routing.py` (`AgentRoutingWorkflow`) -> `examples/workflow/agent_routing.py`
- `pure_tool_workflow.py` (`PureToolWorkflow`) -> `examples/workflow/pure_tool_workflow.py`
- `mixed_agent_workflow.py` (`MixedAgentWorkflow`) -> `examples/workflow/mixed_agent_workflow.py`

## Expected Outputs

- Scripts print structured workflow or runtime result payloads.
- Some flows generate artifacts under `artifacts/examples`.

## Troubleshooting

- Missing local backend dependencies:
  - Install local extras: `pip install -e '.[local]'`.
- LLM startup timeouts:
  - Increase local backend startup timeout in your LLM config if needed.
