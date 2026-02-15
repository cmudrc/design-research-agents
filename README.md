# design-research-agents
[![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Coverage](.github/badges/coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Examples Passing](.github/badges/examples-passing.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Public API In Examples](.github/badges/examples-api-coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)

A modular framework for researching AI agents with shared runtime contracts,
workflow orchestration, and pluggable LLM backends.

## Overview

This project focuses on composable agent systems you can run, inspect, and test:

- Agent implementations (top-level exports): `SingleStepDirectLLMAgent`, `SingleStepRouterAgent`, `SingleStepJsonToolCallingAgent`, `SingleStepCodeToolCallingAgent`, `MultiStepJsonToolCallingAgent`, `MultiStepCodeToolCallingAgent`
- Reusable workflows: `PlanExecuteWorkflow`, `ProposeAndCritiqueWorkflow`, `AgentRoutingWorkflow`, `PureToolWorkflow`, and `MixedAgentWorkflow`
- Workflow orchestration runtime is available via `design_research_agents.workflow` for advanced usage
- Provider-specific LLM clients with constructor-first defaults
- Tracing and structured outputs: consistent metadata, streaming events, and schema-driven payloads

## Public API 
```python
from design_research_agents import (
    LlamaCppServerLLMClient,
    ModelSelectionPolicy,
    PlanExecuteWorkflow,
    SingleStepJsonToolCallingAgent,
    UnifiedToolRuntime,
)

agent = SingleStepJsonToolCallingAgent(...)
tool_runtime = UnifiedToolRuntime(...)
workflow = PlanExecuteWorkflow(...)
policy = ModelSelectionPolicy(...)
llm_client = LlamaCppServerLLMClient()
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
PYTHONPATH=src python3 examples/workflow/plan_execute.py
```

## Examples

See the examples index and sub-guides:

- Top-level examples index: [`examples/README.md`](examples/README.md)
- Agents: [`examples/agents/README.md`](examples/agents/README.md)
- Workflows: [`examples/workflow/README.md`](examples/workflow/README.md)
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
