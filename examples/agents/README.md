## Agent Examples

These examples cover traced agent entrypoints and multi-step execution modes for engineering-design tasks.

## Scripts

- `direct_llm_call.py`
- `multi_step_direct_llm_agent.py`
- `multi_step_json_tool_calling_agent.py`
- `multi_step_code_tool_calling_agent.py`
- `multi_step_json_with_memory.py`

## Quick Start

```bash
PYTHONPATH=src python3 examples/agents/direct_llm_call.py
PYTHONPATH=src python3 examples/agents/multi_step_direct_llm_agent.py
PYTHONPATH=src python3 examples/agents/multi_step_json_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/multi_step_code_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/multi_step_json_with_memory.py
```

## Expected Outputs

- JSON payload including `success`, `final_output`, and `terminated_reason`.
- `trace.trace_path` for each run.
