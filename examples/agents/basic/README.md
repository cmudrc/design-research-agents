## Basic Agent Examples

These examples run one complete non-streaming call per script and print an
`AgentResult`.

## What Each Example Demonstrates

- `single_step_direct_llm_agent.py`
  - One LLM completion without tool calling.
- `single_step_router_agent.py`
  - Route selection across available runtime tools.
- `single_step_json_tool_calling_agent.py`
  - Single-step structured tool selection and invocation.
- `single_step_code_tool_calling_agent.py`
  - Single-step generated-code execution with tool calls.
- `multi_step_code_tool_calling_agent.py`
  - Multi-step ReAct loop using generated code actions.
- `multi_step_json_tool_calling_agent.py`
  - Multi-step ReAct loop using JSON tool-call actions.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/agents/basic/single_step_direct_llm_agent.py
PYTHONPATH=src python3 examples/agents/basic/single_step_router_agent.py
PYTHONPATH=src python3 examples/agents/basic/single_step_json_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/single_step_code_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_code_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_json_tool_calling_agent.py
```

## Expected Outputs

- Each script prints one `AgentResult` payload.
- For tool-calling examples, `tool_results` should contain at least one tool invocation.
- For multi-step examples, metadata should show continuation/iteration behavior.

## Notes

- These examples default to a live local `llama-cpp-server` endpoint.
- Default backend settings come from `dra.llm.create_default_llm_client()`.
- They print `AgentResult` payloads directly for quick inspection.

## Troubleshooting

- Missing local backend dependencies:
  - Install local extras: `pip install -e '.[local]'`.
- Timeouts on first run:
  - Local model server startup can take longer while loading weights.
