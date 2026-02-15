# design-research-agents
[![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Coverage](.github/badges/coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)

A modular framework for researching AI agents with shared runtime contracts,
workflow orchestration, and pluggable LLM backends.

## Overview

This project focuses on composable agent systems you can run, inspect, and test:

- Agent implementations: `dra.agents.SingleStepDirectLLMAgent`, `dra.agents.SingleStepRouterAgent`, `dra.agents.SingleStepJsonToolCallingAgent`, `dra.agents.SingleStepCodeToolCallingAgent`, `dra.agents.MultiStepJsonToolCallingAgent`, `dra.agents.MultiStepCodeToolCallingAgent`
- Unified runtime: `dra.agents.AgentRuntime` modes for `react`, `plan_execute`, `propose_critic`, and `agent_routing`
- Workflow orchestration: `dra.workflows.WorkflowRuntime` with typed logic, tool, and agent steps
- Backend architecture: capability-based routing across local and remote LLM backends
- Tracing and structured outputs: consistent metadata, streaming events, and schema-driven payloads

## Public API 
```python
import design_research_agents as dra

agent = dra.agents.SingleStepJsonToolCallingAgent(...)
tool_runtime = dra.tools.UnifiedToolRuntime(...)
workflow = dra.workflows.WorkflowRuntime(...)
policy = dra.models.ModelSelectionPolicy(...)
router = dra.llm.configure_router_from_yaml("configs/llm.yaml")
```

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
PYTHONPATH=src python3 examples/orchestrator/plan_execute.py
```

## Examples

See the examples index and sub-guides:

- Top-level examples index: [`examples/README.md`](examples/README.md)
- Agents: [`examples/agents/README.md`](examples/agents/README.md)
- Orchestrators: [`examples/orchestrator/README.md`](examples/orchestrator/README.md)
- Model selection: [`examples/model_selection/README.md`](examples/model_selection/README.md)
- Tool runtime source fusion: [`examples/tools/README.md`](examples/tools/README.md)


## Docs

- Getting started: [`docs/quickstart.rst`](docs/quickstart.rst)
- Project philosophy: [`docs/philosophy.rst`](docs/philosophy.rst)
- Agent types: [`docs/agent_types.rst`](docs/agent_types.rst)
- API docs: [`docs/api.rst`](docs/api.rst)

Build docs locally with `make docs`.

## Contributing

Contribution guidelines now live in [`CONTRIBUTING.md`](CONTRIBUTING.md).
