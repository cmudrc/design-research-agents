## Streaming Agent Examples

Run from repo root:

```bash
PYTHONPATH=src python3 examples/agents/streaming/direct_llm_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/router_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/single_step_code_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/multi_step_agent_stream.py
```

Notes:
- `_streaming_support.py` provides deterministic stub clients and event-print helpers.
- These examples do not require a live external model endpoint.
