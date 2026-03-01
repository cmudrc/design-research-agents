# design-research-agents
[![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Examples Passing](https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/examples-passing.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Public API In Examples](https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/examples-api-coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)

`design-research-agents` is a modular framework for prototyping and researching engineering design AI agents.
It features shared runtime contracts, workflow orchestration, and pluggable LLM backends for quick iteration.

## Overview

This library centers on a small set of composable pieces you can run, inspect, and test:

- Two primary entry points: `DirectLLMCall` and `MultiStepAgent` (`direct`, `json`, and `code` modes)
- A tool runtime built around `Toolbox`, with callable, script, and MCP-backed tool configs
- Prebuilt orchestration patterns for plan/execute, debate, propose/critic, routing, beam search, RAG, blackboard, and conversations
- Hosted and local LLM clients, plus `ModelSelector` for backend-selection policies
- Tracing, structured `ExecutionResult` outputs, and runnable examples aimed at repeatable experiments
- A workflow runtime with explicit step primitives for model calls, tool calls, delegation, loops, and memory

## A Super Basic Agent

```python
from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent

with LlamaCppServerLLMClient() as llm_client:
    agent = MultiStepAgent(mode="direct", llm_client=llm_client, max_steps=3)
    result = agent.run(
        prompt="Suggest two design goals for a field-repairable drone battery latch.",
    )

print(result.final_output)
```

## Quickstart

Requires Python 3.12+.
Reproducible release installs are pinned to Python `3.12.12` (see `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate
make dev
make test
PYTHONPATH=src python3 examples/patterns/plan_execute.py
```

For frozen installs, optional extras, and release maintenance, see [Dependencies and Extras](https://cmudrc.github.io/design-research-agents/dependencies_and_extras.html).

## Examples

Start with [examples/README.md](https://github.com/cmudrc/design-research-agents/blob/main/examples/README.md) for runnable examples grouped by agents, clients, workflows, patterns, model selection, and tools.


## Docs

See the [documentation site](https://cmudrc.github.io/design-research-agents/) for the full guide set, including quickstart, backend setup, workflows, patterns, and API reference.

Build docs locally with `make docs`.

## Contributing

Contribution guidelines now live in [CONTRIBUTING.md](https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md).
