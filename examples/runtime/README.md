## AgentRuntime Examples

These scripts demonstrate `AgentRuntime` execution modes.

Run from repo root:

```bash
PYTHONPATH=src python3 examples/runtime/plan_execute.py
PYTHONPATH=src python3 examples/runtime/propose_critic.py
PYTHONPATH=src python3 examples/runtime/triage.py
```

Notes:
- `_runtime_example_support.py` provides deterministic sequence-based LLM stubs.
- `react` mode is represented by `MultiStepAgent` semantics in the runtime implementation.
