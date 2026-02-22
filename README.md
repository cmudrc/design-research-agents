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

- Agent implementations (top-level exports): `DirectLLMCall`, `MultiStepAgent` (modes: `direct`, `json`, `code`)
- Workflow construction surface: `design_research_agents.workflow` (`Workflow` + step primitives)
- Prebuilt workflow implementations: `design_research_agents.patterns` (`DebatePattern`, `PlannerExecutorPattern`, `ReflexionPattern`, `RouterPattern`, etc.)
- Provider-specific LLM clients with constructor-first defaults
- Tracing and structured outputs: consistent metadata and schema-driven payloads

## Public API 
```python
from design_research_agents import (
    AgentStep,
    DirectLLMCall,
    LlamaCppServerLLMClient,
    LoopStep,
    ModelSelector,
    PlannerExecutorPattern,
    MultiStepAgent,
    Workflow,
    Toolbox,
)
from design_research_agents.patterns import PlannerExecutorPattern

agent = MultiStepAgent(mode="json", ...)
direct = DirectLLMCall(...)
tool_runtime = Toolbox(...)
pattern = PlannerExecutorPattern(...)
workflow = Workflow(steps=(AgentStep(...), LoopStep(...)))
selector = ModelSelector(...)
decision = selector.select(task="summarize interview findings", output="decision")
llm_client = LlamaCppServerLLMClient()
```

## Quickstart

Requires Python 3.12+.
Reproducible release installs are pinned to Python `3.12.12` (see `.python-version`).

### Normal install (library development)

```bash
python -m venv .venv
source .venv/bin/activate
make dev
make test
make run-example
```

### Reproducible install (frozen)

`make repro` uses `uv.lock` in frozen mode and fails if the lock is out of date.

```bash
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
make repro REPRO_EXTRAS="dev full"
make test
```

`REPRO_EXTRAS` defaults to `dev full`.

Example run:

```bash
PYTHONPATH=src python3 examples/patterns/plan_execute.py
```

## Reproducible release process (maintainers)

On each release:

1. Use Python `3.12.12` (the pinned release interpreter in `.python-version`).
2. Regenerate lock data: `make lock`.
3. Verify frozen install + tests: `make repro REPRO_EXTRAS="dev full"` and `make ci`.
4. Commit `uv.lock` (and any dependency spec updates), then tag and publish the release.

## Examples

See the examples index and sub-guides:

- Top-level examples index: [`examples/README.md`](examples/README.md)
- Agents: [`examples/agents/README.md`](examples/agents/README.md)
- Client configuration: [`examples/clients/README.md`](examples/clients/README.md)
- Workflow primitives: [`examples/workflow/README.md`](examples/workflow/README.md)
- Patterns: [`examples/patterns/README.md`](examples/patterns/README.md)
- Model selection: [`examples/model_selection/README.md`](examples/model_selection/README.md)
- Tool runtime + script tools: [`examples/tools/README.md`](examples/tools/README.md)


## Docs

- Getting started: [`docs/quickstart.rst`](docs/quickstart.rst)
- Dependencies + extras: [`docs/dependencies_and_extras.rst`](docs/dependencies_and_extras.rst)
- Project philosophy: [`docs/philosophy.rst`](docs/philosophy.rst)
- LLM clients: [`docs/llm_clients/index.rst`](docs/llm_clients/index.rst)
- Agents: [`docs/agents/index.rst`](docs/agents/index.rst)
- Tools: [`docs/tools/index.rst`](docs/tools/index.rst)
- Workflow builders: [`docs/workflows/index.rst`](docs/workflows/index.rst)
- Patterns: [`docs/patterns/index.rst`](docs/patterns/index.rst)
- API docs: [`docs/api.rst`](docs/api.rst)

Build docs locally with `make docs`.

## Contributing

Contribution guidelines now live in [`CONTRIBUTING.md`](CONTRIBUTING.md).
