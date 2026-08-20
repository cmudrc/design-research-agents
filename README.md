# design-research-agents
[![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
[![Examples Passing](https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/examples-passing.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml)
[![API in Examples](https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/examples-api-coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml)
[![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)
[![PyPI Version](https://img.shields.io/pypi/v/design-research-agents.svg)](https://pypi.org/project/design-research-agents/)
[![Python Versions](https://img.shields.io/pypi/pyversions/design-research-agents.svg)](https://pypi.org/project/design-research-agents/)

`design-research-agents` is the agent-execution layer in the
CMU Design Research Collective design-research ecosystem. It owns executable AI
participants, workflow and tool runtimes, model-client adapters, and traceable
run results.

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

This package centers on reproducible agent workflows with a layered public API:

- Two primary entry points: `DirectLLMCall` and `MultiStepAgent` (`direct`, `json`, and `code` modes)
- A seeded random control-condition agent for packaged-problem studies (`SeededRandomBaselineAgent`)
- A prompt-driven workflow agent for packaged-problem studies (`PromptWorkflowAgent`)
- A study-facing execution facade in `design_research_agents.study` for experiment runners
- Workflow primitives for model, tool, delegate, loop, and memory steps
- A tool runtime built around `Toolbox`, with callable, script, and MCP-backed tool configs
- Hosted and local LLM clients, model flights/catalogs, and `ModelSelector` for backend-selection policies
- Prebuilt coordination and reasoning patterns for plan/execute, propose/critic, debate, routing, round-based coordination, blackboard, tree search, Ralph loops, nominal teams, RAG, and conversation
- Tracing, structured `ExecutionResult` outputs, and runnable examples aimed at repeatable experiments

## A Super Basic Agent

The first example is deliberately offline and deterministic. It exercises
`DirectLLMCall` through the minimal runtime methods that this participant
uses, without downloading a model, using an API key, or starting a model
server. The stub is intentionally smaller than the complete `LLMClient`
interface implemented by the packaged, type-checked backends.

```python
from design_research_agents import DirectLLMCall, LLMRequest, LLMResponse


class LocalStubClient:
    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=f"Offline response to: {request.messages[-1].content}",
            model="local-stub",
            provider="local-stub",
        )

    def default_model(self) -> str:
        return "local-stub"

    def close(self) -> None:
        return None


agent = DirectLLMCall(llm_client=LocalStubClient())
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

Start with the published base install, then save the offline snippet above as
``offline_agent.py`` and run it:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install design-research-agents
python offline_agent.py
```

For contributor work, clone the repository before using its Make targets and
checked-in example:

```bash
git clone https://github.com/cmudrc/design-research-agents.git
cd design-research-agents
python -m venv .venv
source .venv/bin/activate
make dev
make test
PYTHONPATH=src python examples/agents/vscode_hello_world.py
```

That checked-in onboarding example uses a deterministic local stub, so the base
install needs no model service or provider credentials. Real backend runtimes
are explicit extras. Use
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

For a zero-service first run, use
`examples/agents/vscode_hello_world.py`. Examples that use a hosted client,
llama.cpp, Ollama, or another local server state their backend prerequisites.

Some local `LlamaCppServerLLMClient` examples intentionally use `Qwen3-4B`
GGUF configs, which can exceed available RAM on smaller machines. If you want a
lighter local starting point, begin with the
[Ollama local client docs](https://cmudrc.github.io/design-research-agents/examples/clients/ollama_local_client.html)
or the
[OllamaLLMClient guide](https://cmudrc.github.io/design-research-agents/llm_clients/ollama_local.html).

## Docs

See the [published documentation](https://cmudrc.github.io/design-research-agents/)
for quickstart guidance, backend setup, workflow/pattern guides, and API docs.
The [Guides](https://cmudrc.github.io/design-research-agents/guides.html) page
provides the shared install → quickstart → concepts/workflow → examples → API
path used across the ecosystem.

Check generated documentation consistency, then run the strict build:

```bash
make docs-check
make docs-build
```

## Ecosystem Role and Compatibility

This package executes participants; it does not own benchmark definitions,
study design, or downstream interpretation. Use the sibling layers for those
responsibilities:

- [design-research-problems](https://cmudrc.github.io/design-research-problems/) owns benchmark tasks, metadata, and evaluators.
- [design-research-experiments](https://cmudrc.github.io/design-research-experiments/) owns study design and orchestration across packages.
- [design-research-analysis](https://cmudrc.github.io/design-research-analysis/) owns validation and analysis of exported study records.

Compatibility is guaranteed for the curated top-level `__all__` surface and
the documented public facade modules. `design_research_agents.study` is the
preferred study-facing facade; `design_research_agents.integration` remains a
compatibility module for existing orchestration consumers. See the
[API reference](https://cmudrc.github.io/design-research-agents/api.html) for
the exact boundary and the umbrella
[compatibility matrix](https://cmudrc.github.io/design-research/compatibility.html)
for the component versions tested together.

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
- Patterns: conversation, debate, plan/execute, propose/critic, Ralph loops, nominal teams, routing, round-based coordination, blackboard, tree search, simulated annealing, and RAG
- LLM clients: hosted and local adapters, including OpenAI-compatible HTTP plus provider-specific clients
- Runtime services: `design_research_agents.model_selection`, `ModelFlightRegistry`,
  `ModelCatalog`, `ModelSelector`, and `Tracer`

## Contributing

Contribution workflow and quality gates are documented in
[CONTRIBUTING.md](https://github.com/cmudrc/design-research-agents/blob/HEAD/CONTRIBUTING.md).
