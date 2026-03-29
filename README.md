# design-research-agents
[![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Examples Passing](https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/examples-passing.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Public API In Examples](https://raw.githubusercontent.com/cmudrc/design-research-agents/main/.github/badges/examples-api-coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)

`design-research-agents` is the agent-execution layer in the cmudrc design
research ecosystem.

It provides typed, composable contracts for direct calls, multi-step runs,
workflow orchestration, tool execution, and traceable experimentation.

## Overview

This package centers on reproducible agent workflows with a compact public API:

- Direct and multi-step agent entry points (`DirectLLMCall`, `MultiStepAgent`)
- Workflow primitives for model, tool, delegate, loop, and memory steps
- A unified tool runtime (`Toolbox`) for callable, script, and MCP-backed tools
- Hosted and local LLM clients, plus `ModelSelector` policy-driven backend selection
- Prebuilt coordination patterns for plan/execute, propose/critic, debate, routing, beam search, RAG, blackboard, and conversation
- Tracing hooks and structured execution outputs for deterministic evaluation

## Quickstart

Requires Python 3.12+.
Reproducible release installs are pinned to Python `3.12.12` (`.python-version`).

If you prefer a guided editor-first flow, use the
[VS Code Setup Guide](https://cmudrc.github.io/design-research-agents/vscode_setup.html).
It walks through creating a virtual environment, installing the published
package, and running a first script in VS Code.

```bash
python -m venv .venv
source .venv/bin/activate
make dev
make test
PYTHONPATH=src python examples/agents/direct_llm_call.py
```

The base-install path uses `OpenAICompatibleHTTPLLMClient` and expects a running
OpenAI-compatible endpoint. Contributor setup (`make dev`) installs development
tooling only; backend runtimes are explicit extras.

For frozen installs, extras, and release maintenance, see
[Dependencies and Extras](https://cmudrc.github.io/design-research-agents/dependencies_and_extras.html).

## Examples

Start with [examples/README.md](https://github.com/cmudrc/design-research-agents/blob/main/examples/README.md)
for runnable examples grouped by agents, clients, workflows, patterns, model
selection, and tools.

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

- Agent entry points: `DirectLLMCall`, `MultiStepAgent`
- Core contracts: `ExecutionResult`, `LLMRequest`, `LLMMessage`, `LLMResponse`, `ToolResult`
- Workflow runtime: `Workflow`, `CompiledExecution`, and step contracts for model/tool/delegate/loop/memory behavior
- Tools: `Toolbox`, `CallableToolConfig`, `ScriptToolConfig`, `MCPServerConfig`
- Patterns: conversation, debate, plan/execute, propose/critic, routing, round-based coordination, blackboard, beam search, and RAG
- LLM clients: hosted and local adapters, including OpenAI-compatible HTTP plus provider-specific clients
- Runtime services: `ModelSelector` and `Tracer`

## Contributing

Contribution workflow and quality gates are documented in
[CONTRIBUTING.md](https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md).
