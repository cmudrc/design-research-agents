## Agent Examples

These examples cover traced agent entrypoints and multi-step execution modes for engineering-design tasks.

## Scripts

- `direct_llm_call.py`
- `direct_llm_compiled_execution.py`
- `multi_step_direct_llm_agent.py`
- `multi_step_json_tool_calling_agent.py`
- `multi_step_code_tool_calling_agent.py`
- `multi_step_json_with_memory.py`
- `seeded_random_baseline_agent.py`
- `prompt_workflow_agent.py`

## Quick Start

```bash
PYTHONPATH=src python3 examples/agents/direct_llm_call.py
PYTHONPATH=src python3 examples/agents/direct_llm_compiled_execution.py
PYTHONPATH=src python3 examples/agents/multi_step_direct_llm_agent.py
PYTHONPATH=src python3 examples/agents/multi_step_json_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/multi_step_code_tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/multi_step_json_with_memory.py
PYTHONPATH=src python3 examples/agents/seeded_random_baseline_agent.py
PYTHONPATH=src python3 examples/agents/prompt_workflow_agent.py
```

## Expected Outputs

- Most model-backed examples print an `ExecutionResult.summary()` JSON payload including `success`,
  `final_output`, and `terminated_reason`.
- `seeded_random_baseline_agent.py` uses the same `run(prompt, dependencies=...)` contract as the
  other public agents, then prints a deterministic comparison payload for a random control condition
  versus a non-random baseline.
- `prompt_workflow_agent.py` shows how `PromptWorkflowAgent` converts packaged-problem study
  metadata into a prompt-mode workflow run while preserving a stable `request_id`.
- Trace paths are included for the model-backed runs.
