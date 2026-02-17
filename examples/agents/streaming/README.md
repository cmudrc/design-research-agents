## Streaming Agent Examples

These examples run streaming variants of the core agents. Each script emits
incremental events and then a completed result.

## What Each Example Demonstrates

- `single_step_direct_llm_agent_stream.py`
  - Streaming text-only generation.
- `single_step_tool_router_agent_stream.py`
  - Streaming tool-router behavior.
- `single_step_json_tool_calling_agent_stream.py`
  - Streaming single-step JSON tool selection and execution.
- `single_step_code_tool_calling_agent_stream.py`
  - Streaming single-step code agent execution.
- `multi_step_code_tool_calling_agent_stream.py`
  - Streaming multi-step code-driven ReAct loop.
- `multi_step_json_tool_calling_agent_stream.py`
  - Streaming multi-step JSON-driven ReAct loop.
- `multi_step_tool_router_agent_stream.py`
  - Streaming multi-step TOOL_CALL/STOP routing loop.
- `multi_step_direct_llm_agent_stream.py`
  - Streaming multi-step direct-response CONTINUE/STOP loop.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/agents/streaming/single_step_direct_llm_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/single_step_tool_router_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/single_step_json_tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/single_step_code_tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/multi_step_code_tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/multi_step_json_tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/multi_step_tool_router_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/multi_step_direct_llm_agent_stream.py
```

## Expected Outputs

- Console output should include one or more `delta` events.
- A final `completed` event should include a successful result payload.

## Notes

- These examples default to a live local `llama-cpp-server` endpoint.
- Default backend settings come from `LlamaCppServerLLMClient()`.

## Troubleshooting

- Missing local backend dependencies:
  - Install local extras: `pip install -e '.[local]'`.
- Slow or empty streaming output:
  - Confirm the local model server is reachable and has finished startup.
