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
from design_research_agents import DirectLLMCall, HTMLLLMClient

with HTMLLLMClient() as llm_client:
    agent = DirectLLMCall(llm_client=llm_client)
    result = agent.run("Suggest two design goals for a field-repairable drone battery latch.")

print(result.final_output)
```

`HTMLLLMClient` is the built-in minimal-install stand-in. For the main cloud and
local paths most users will want, see the Quickstart guide for
`OpenAIServiceLLMClient` and `LlamaCppServerLLMClient`.

## Quickstart

Requires Python 3.12+.
Reproducible release installs are pinned to Python `3.12.12` (see `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate
make dev
make test
PYTHONPATH=src python3 examples/agents/direct_llm_call.py
```

That example is the minimal base-install path. ``make dev`` installs contributor
tooling only; backend runtimes remain explicit extras. Use ``.[openai]`` for
OpenAI, ``.[azure]`` for Azure OpenAI (same SDK, explicit intent), ``.[llama_cpp]``
for the recommended local path, ``.[providers]`` for all hosted provider SDKs,
or ``.[full]`` for hosted + local coverage. The full quickstart guide also
covers the main OpenAI (cloud) and llama.cpp (local) setups.

For frozen installs, optional extras, and release maintenance, see [Dependencies and Extras](https://cmudrc.github.io/design-research-agents/dependencies_and_extras.html).

## Examples

Start with [examples/README.md](https://github.com/cmudrc/design-research-agents/blob/main/examples/README.md) for runnable examples grouped by agents, clients, workflows, patterns, model selection, and tools.


## Docs

See the [documentation site](https://cmudrc.github.io/design-research-agents/) for the full guide set, including quickstart, backend setup, workflows, patterns, and API reference.

Build docs locally with `make docs`.

## Contributing

Contribution guidelines now live in [CONTRIBUTING.md](https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md).
