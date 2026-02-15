## Examples

This directory is organized by capability so you can find runnable scripts quickly.

- `examples/agents`
  - Core agent examples.
  - See `examples/agents/README.md`.
- `examples/runtime`
  - Unified `AgentRuntime` mode examples.
  - See `examples/runtime/README.md`.
- `examples/orchestrator`
  - Sequential and DAG workflow examples.
  - See `examples/orchestrator/README.md`.
- `examples/model_selection`
  - Local/remote model selection policy examples.
  - See `examples/model_selection/README.md`.
- `examples/tools`
  - Unified tool runtime source-fusion examples (core + lazy + MCP).
  - See `examples/tools/README.md`.

All scripts are intended to be run from repository root with `PYTHONPATH=src`.
Agent/runtime/orchestrator examples default to local llama-cpp-server via
`dra.llm.create_default_llm_client()`.

Example:

```bash
PYTHONPATH=src python3 examples/agents/basic/single_step_direct_llm_agent.py
```
