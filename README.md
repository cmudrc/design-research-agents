# design-research-agents
[![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Examples Passing](https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/examples-passing.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml)
[![API in Examples](https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/examples-api-coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml)
[![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)
[![PyPI Version](https://img.shields.io/pypi/v/design-research-agents.svg)](https://pypi.org/project/design-research-agents/)
[![Python Versions](https://img.shields.io/pypi/pyversions/design-research-agents.svg)](https://pypi.org/project/design-research-agents/)

`design-research-agents` is the agent-execution layer in the cmudrc design
research ecosystem.

It provides typed, composable contracts for direct calls, multi-step runs,
workflow orchestration, tool execution, and traceable experimentation.

If you are deciding between primitives, workflow authoring, prebuilt patterns,
and runnable exemplars, start with the
[Where To Start](https://cmudrc.github.io/design-research-agents/where_to_start.html)
guide in the published docs.

## Quality Signals

- **Coverage** reports total line coverage for the default deterministic test suite; CI requires at least 95%.
- **Examples Passing** reports checked-in example scripts that execute successfully in the examples workflow.
- **API in Examples** reports curated top-level `__all__` exports referenced by runnable examples. `N/N` means every supported top-level export appears in at least one example, and CI requires 100%.

Run `make coverage`, `make examples-test`, and `make examples-coverage` to reproduce these checks locally.

## Overview

This package centers on reproducible agent workflows with a compact public API:

- Two primary entry points: `DirectLLMCall` and `MultiStepAgent` (`direct`, `json`, and `code` modes)
- A seeded random control-condition agent for packaged-problem studies (`SeededRandomBaselineAgent`)
- A prompt-driven workflow agent for packaged-problem studies (`PromptWorkflowAgent`)
- A study-facing execution facade in `design_research_agents.study` for experiment runners
- Workflow primitives for model, tool, delegate, loop, and memory steps
- A tool runtime built around `Toolbox`, with callable, script, and MCP-backed tool configs
- Hosted and local LLM clients, model flights/catalogs, and `ModelSelector` for backend-selection policies
- Prebuilt patterns for coordination, reasoning, tree search, simulated annealing,
  and reinforcement learning
- Tracing, structured `ExecutionResult` outputs, and runnable examples aimed at repeatable experiments

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
Reproducible release installs target Python `3.12` (see `.python-version`).

On Windows, if `python` or `pip` resolve to an older interpreter, use
`py -3.12 -m venv .venv` and `py -3.12 -m pip ...` for the environment-creation
and package-install steps.

If you prefer a guided editor-first flow, use the
[VS Code Setup Guide](https://cmudrc.github.io/design-research-agents/vscode_setup.html).
It walks through creating a virtual environment, installing the published
package, running a first script in VS Code, and using the source checkout for
repository examples.

```bash
python3 -m venv .venv
source .venv/bin/activate
make dev
make test
PYTHONPATH=src python examples/agents/direct_llm_call.py
```

The base-install path uses `OpenAICompatibleHTTPLLMClient` and expects a running
OpenAI-compatible endpoint. Contributor setup (`make dev`) installs development
tooling only; backend runtimes are explicit extras. Use
`design-research-agents[full]` for the hosted + local backend bundle and
`design-research-agents[all]` when you also want the optional ChromaDB and
graph-memory backends. Use `design-research-agents[huggingface]` when you only
need Hugging Face Hub metadata for catalog discovery.

For frozen installs, extras, and release maintenance, see
[Dependencies and Extras](https://cmudrc.github.io/design-research-agents/dependencies_and_extras.html).

## Examples

Start with [examples/README.md](https://github.com/cmudrc/design-research-agents/blob/HEAD/examples/README.md)
for runnable examples grouped by agents, clients, workflows, patterns, model
selection, and tools.

Some local `LlamaCppServerLLMClient` examples intentionally use `Qwen3-4B`
GGUF configs, which can exceed available RAM on smaller machines. If you want a
lighter local starting point, begin with the
[Ollama local client docs](https://cmudrc.github.io/design-research-agents/examples/clients/ollama_local_client.html)
or the
[OllamaLLMClient guide](https://cmudrc.github.io/design-research-agents/llm_clients/ollama_local.html).

## Docs

See the [published documentation](https://cmudrc.github.io/design-research-agents/)
for quickstart guidance, backend setup, workflow/pattern guides, and API docs.

Build docs locally with:

```bash
make docs
```

## Public API

The supported public surface is whatever is exported from
`design_research_agents.__all__`.

Top-level exports include:

- Agent entry points: `DirectLLMCall`, `MultiStepAgent`, `SeededRandomBaselineAgent`, `PromptWorkflowAgent`
- Study-facing helpers: the `study` module, `AgentRunRequest`, `execute_agent_request`,
  `execute_agent_run`, and `normalize_agent_execution`
- Core contracts: `ExecutionResult`, `LLMRequest`, `LLMMessage`, `LLMResponse`, `ToolResult`
- Workflow runtime: `Workflow`, `CompiledExecution`, and step contracts for model/tool/delegate/loop/memory behavior
- Tools: `Toolbox`, `CallableToolConfig`, `ScriptToolConfig`, `MCPServerConfig`
- Patterns: conversation, debate, plan/execute, propose/critic, Ralph loops,
  nominal teams, routing, round-based coordination, blackboard, tree search,
  RAG, simulated annealing, and reinforcement learning
- LLM clients: hosted and local adapters, including OpenAI-compatible HTTP plus provider-specific clients
- Runtime services: `design_research_agents.model_selection`, `ModelFlightRegistry`,
  `ModelCatalog`, `ModelSelector`, and `Tracer`

## Contributing

Contribution workflow and quality gates are documented in
[CONTRIBUTING.md](https://github.com/cmudrc/design-research-agents/blob/HEAD/CONTRIBUTING.md).
