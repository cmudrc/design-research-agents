## Examples

This directory contains runnable end-to-end examples for the major framework
surfaces: agents, clients, orchestration workflows, model selection, and tool runtime.

## Directory Guide

- `examples/agents`
  - Core agent patterns (single-step, multi-step, streaming).
  - See `examples/agents/README.md`.
- `examples/clients`
  - Purposeful client-configuration examples for each supported LLM client type.
  - See `examples/clients/README.md`.
- `examples/workflow`
  - Reusable orchestration entrypoints and workflow runtime integrations.
  - See `examples/workflow/README.md`.
- `examples/model_selection`
  - Local-vs-remote model selection policy behavior.
  - See `examples/model_selection/README.md`.
- `examples/optimization`
  - One-dimensional optimization examples using ToolRouting and tool-calling agents.
  - See `examples/optimization/README.md`.
- `examples/tools`
  - Unified tool runtime examples (core + script + MCP), including script-tool wrappers.
  - See `examples/tools/README.md`.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/agents/basic/single_step_direct_llm_agent.py
```

## Expected Outputs

- Most examples print JSON-like result payloads (`AgentResult`, workflow result,
  or tool runtime report).
- Streaming examples print incremental events (`delta`) followed by completion.
- Tool and script-tool examples also write artifacts under `artifacts/`.

## Troubleshooting

- `ModuleNotFoundError`:
  - Run with `PYTHONPATH=src` from the repo root.
- Local LLM startup issues:
  - Install local extras: `pip install -e '.[local]'`.
- Slow first run:
  - Local model server startup and model loading can take additional time.
