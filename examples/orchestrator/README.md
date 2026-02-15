## Workflow Runtime Examples

Run from repo root:

```bash
PYTHONPATH=src python3 examples/orchestrator/pure_tool_workflow.py
PYTHONPATH=src python3 examples/orchestrator/mixed_agent_workflow.py
```

Notes:
- `pure_tool_workflow.py` shows pure workflow execution with chained tool and logic steps.
- `mixed_agent_workflow.py` shows one workflow that combines logic routing, agent delegation, and tool execution.
