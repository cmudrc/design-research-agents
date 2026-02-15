## AgentRuntime Examples

These scripts demonstrate `AgentRuntime` execution modes.

Run from repo root:

```bash
PYTHONPATH=src python3 examples/runtime/plan_execute.py
PYTHONPATH=src python3 examples/runtime/propose_critic.py
PYTHONPATH=src python3 examples/runtime/triage.py
```

Notes:
- These examples default to a live local `llama-cpp-server` endpoint.
- Default backend settings come from `dra.llm.create_default_llm_client()`.
- `react` mode is represented by `MultiStepCodeToolCallingAgent` semantics in the runtime implementation.
