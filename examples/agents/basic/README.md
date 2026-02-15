## Basic Agent Examples

Run from repo root:

```bash
PYTHONPATH=src python3 examples/agents/basic/single_step_direct_llm_agent.py
PYTHONPATH=src python3 examples/agents/basic/single_step_router_agent.py
PYTHONPATH=src python3 examples/agents/basic/single_step_json_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/single_step_code_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_code_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_json_tool_calling_agent.py
```

Notes:
- These examples default to a live local `llama-cpp-server` endpoint.
- Default backend settings come from `dra.llm.create_default_llm_client()`.
- They print `AgentResult` payloads directly for quick inspection.
