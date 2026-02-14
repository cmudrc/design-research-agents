# design-research-agents

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
PYTHONPATH=src python3 examples/router_agent.py
PYTHONPATH=src python3 examples/tool_calling_agent.py
PYTHONPATH=src python3 examples/single_step_code_agent.py
PYTHONPATH=src python3 examples/multi_step_agent.py
```

Default backend is `llama-cpp-server`; examples use hardcoded local llama-cpp settings.

## Docs

- Getting started: [`docs/quickstart.rst`](docs/quickstart.rst)
- Project philosophy: [`docs/philosophy.rst`](docs/philosophy.rst)
- Agent types: [`docs/agent_types.rst`](docs/agent_types.rst)
- API docs: [`docs/api.rst`](docs/api.rst)

Build docs locally with `make docs`.

## Contributing

Contribution guidelines now live in [`CONTRIBUTING.md`](CONTRIBUTING.md).
