## Streaming Agent Examples

Run from repo root:

```bash
PYTHONPATH=src python3 examples/agents/streaming/single_step_direct_llm_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/single_step_router_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/single_step_json_tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/single_step_code_tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/multi_step_code_tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/multi_step_json_tool_calling_agent_stream.py
```

Notes:
- These examples default to a live local `llama-cpp-server` endpoint.
- Default backend settings come from `dra.llm.create_default_llm_client()`.
