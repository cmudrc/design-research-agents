## Workflow Runtime Examples

These entrypoints exercise orchestration-level flows, including reusable
workflow chunks in `src/design_research_agents/orchestrator/implementations/`.
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
PYTHONPATH=src python3 examples/orchestrator/workflow_runtime.py
PYTHONPATH=src python3 examples/orchestrator/plan_execute.py
PYTHONPATH=src python3 examples/orchestrator/propose_critic.py
PYTHONPATH=src python3 examples/orchestrator/agent_routing.py
PYTHONPATH=src python3 examples/orchestrator/pure_tool_workflow.py
PYTHONPATH=src python3 examples/orchestrator/mixed_agent_workflow.py
```

## Implementation Mapping

- `workflow_runtime.py` (`WorkflowRuntime`) -> `examples/orchestrator/workflow_runtime.py`
- `plan_execute.py` (`plan_and_execute`) -> `examples/orchestrator/plan_execute.py`
- `propose_critic.py` (`propose_and_critique`) -> `examples/orchestrator/propose_critic.py`
- `agent_routing.py` (`agent_routing_and_delegate`) -> `examples/orchestrator/agent_routing.py`
- `pure_tool_workflow.py` (`pure_tool_workflow`) -> `examples/orchestrator/pure_tool_workflow.py`
- `mixed_agent_workflow.py` (`mixed_agent_workflow`) -> `examples/orchestrator/mixed_agent_workflow.py`

## Expected Outputs

- Scripts print structured workflow or runtime result payloads.
- Some flows generate artifacts under `artifacts/examples`.

## Troubleshooting

- Missing local backend dependencies:
  - Install local extras: `pip install -e '.[local]'`.
- LLM startup timeouts:
  - Increase local backend startup timeout in your LLM config if needed.
