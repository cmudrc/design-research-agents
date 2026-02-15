## Basic Agent Examples

Run from repo root:

```bash
PYTHONPATH=src python3 examples/agents/basic/direct_llm_agent.py
PYTHONPATH=src python3 examples/agents/basic/router_agent.py
PYTHONPATH=src python3 examples/agents/basic/tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/single_step_code_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_agent.py
```

Notes:
- These examples use package defaults (including the default local llama-cpp client in many cases).
- They print `AgentResult` payloads directly for quick inspection.
