# design-research-agents
[![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)

A flexible, modular framework for researching AI agents that design

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,local]"
make test
make run-example
```

Run additional basic examples:

```bash
PYTHONPATH=src python3 examples/agents/basic/router_agent.py
PYTHONPATH=src python3 examples/agents/basic/direct_llm_agent.py
PYTHONPATH=src python3 examples/agents/basic/tool_calling_agent.py
PYTHONPATH=src python3 examples/agents/basic/single_step_code_agent.py
PYTHONPATH=src python3 examples/agents/basic/multi_step_agent.py
PYTHONPATH=src python3 examples/runtime/plan_execute.py
PYTHONPATH=src python3 examples/runtime/propose_critic.py
PYTHONPATH=src python3 examples/runtime/triage.py
PYTHONPATH=src python3 examples/orchestrator/sequential.py
PYTHONPATH=src python3 examples/orchestrator/dag.py
PYTHONPATH=src python3 examples/orchestrator/research_pipeline_dag.py
PYTHONPATH=src python3 examples/model_selection/local.py
PYTHONPATH=src python3 examples/model_selection/remote.py
```

Run additional streaming examples:

```bash
PYTHONPATH=src python3 examples/agents/streaming/direct_llm_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/router_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/tool_calling_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/single_step_code_agent_stream.py
PYTHONPATH=src python3 examples/agents/streaming/multi_step_agent_stream.py
```

These streaming examples use deterministic in-script LLM stubs and do not
require a running model backend.

Default backend is `llama-cpp-server`; examples use hardcoded local llama-cpp settings.
Optional local backend: `transformers` (install `transformers` + a torch runtime).

## Docs

- Getting started: [`docs/quickstart.rst`](docs/quickstart.rst)
- Project philosophy: [`docs/philosophy.rst`](docs/philosophy.rst)
- Agent types: [`docs/agent_types.rst`](docs/agent_types.rst)
- API docs: [`docs/api.rst`](docs/api.rst)

Build docs locally with `make docs`.

## Contributing

Contribution guidelines now live in [`CONTRIBUTING.md`](CONTRIBUTING.md).
