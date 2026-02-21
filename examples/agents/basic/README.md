## Basic Agent Examples

These scripts run one complete non-streaming call per example and print compact
result payloads with trace metadata.

## Examples

- `direct_llm_call.py`
  - `DirectLLMCall` for one engineering-design objective prompt.
- `multi_step_direct_llm_agent.py`
  - `MultiStepAgent(mode="direct")` with CONTINUE/STOP progression.
- `multi_step_json_tool_calling_agent.py`
  - `MultiStepAgent(mode="json")` with callable-tool risk scoring.
- `multi_step_code_tool_calling_agent.py`
  - `MultiStepAgent(mode="code")` with tool-backed code action steps.
- `multi_step_json_with_memory.py`
  - `MultiStepAgent(mode="json")` with memory retrieval/write-back.

## Quick Start

```bash
PYTHONPATH=src python3 examples/agents/basic/direct_llm_call.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_direct_llm_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_json_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_code_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_json_with_memory.py
```

## Expected Outputs

- `success`, termination/final output fields, and run summary metadata.
- `trace.trace_path` for each run.
