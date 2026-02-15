# design-research-agents
[![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)

A modular framework for researching AI agents with shared runtime contracts,
workflow orchestration, and pluggable LLM backends.

## Overview

This project focuses on composable agent systems you can run, inspect, and test:

- Agent implementations: `DirectLLMAgent`, `ToolCallingAgent`, `SingleStepCodeAgent`, `MultiStepAgent`
- Unified runtime: `AgentRuntime` modes for `react`, `plan_execute`, `propose_critic`, and `triage`
- Workflow orchestration: `WorkflowRuntime` with typed logic, tool, and agent steps
- Backend architecture: capability-based routing across local and remote LLM backends
- Tracing and structured outputs: consistent metadata, streaming events, and schema-driven payloads

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,local]"
make test
make run-example
```

Example run:

```bash
PYTHONPATH=src python3 examples/runtime/plan_execute.py
```

## Examples

See the examples index and sub-guides:

- Top-level examples index: [`examples/README.md`](examples/README.md)
- Agents: [`examples/agents/README.md`](examples/agents/README.md)
- Runtime modes: [`examples/runtime/README.md`](examples/runtime/README.md)
- Orchestrators: [`examples/orchestrator/README.md`](examples/orchestrator/README.md)
- Model selection: [`examples/model_selection/README.md`](examples/model_selection/README.md)

Most examples run with deterministic in-repo stubs. Backends like
`llama_cpp`, `transformers_local`, and `mlx_local` are available when optional
dependencies are installed.

## Docs

- Getting started: [`docs/quickstart.rst`](docs/quickstart.rst)
- Project philosophy: [`docs/philosophy.rst`](docs/philosophy.rst)
- Agent types: [`docs/agent_types.rst`](docs/agent_types.rst)
- API docs: [`docs/api.rst`](docs/api.rst)

Build docs locally with `make docs`.

## Contributing

Contribution guidelines now live in [`CONTRIBUTING.md`](CONTRIBUTING.md).
