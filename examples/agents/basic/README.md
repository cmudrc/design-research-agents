## Basic Agent Examples

These examples run one complete non-streaming call per script and print an
`ExecutionResult`.

## What Each Example Demonstrates

- `direct_llm_call.py`
  - One direct LLM completion without tool calling.
- `multi_step_code_tool_calling_agent.py`
  - Multi-step ReAct loop using generated code actions.
- `multi_step_json_tool_calling_agent.py`
  - Multi-step ReAct loop using JSON tool-call actions.
- `multi_step_json_with_memory.py`
  - Multi-step JSON tool-calling with local memory retrieval and write-back.
- `multi_step_tool_router_agent.py`
  - Multi-step ReAct loop where each step is TOOL_CALL or STOP.
- `multi_step_direct_llm_agent.py`
  - Multi-step direct-response controller with CONTINUE/STOP decisions.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/agents/basic/direct_llm_call.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_code_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_json_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_json_with_memory.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_tool_router_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_direct_llm_agent.py
```

## Expected Outputs

- Each script prints one `ExecutionResult` payload.
- For tool-calling examples, `tool_results` should contain at least one tool invocation.
- For multi-step examples, metadata should show continuation/iteration behavior.

## Notes

- Most examples default to a live local `llama-cpp-server` endpoint.
- They print `ExecutionResult` payloads directly for quick inspection.

## Troubleshooting

- Missing local backend dependencies:
  - Install local extras: `pip install -e '.[local]'`.
- Timeouts on first run:
  - Local model server startup can take longer while loading weights.
