# Examples

Runnable examples for `design-research-agents`.

## Directory Guide

- `agents/`: direct and multi-step agent entry points.
- `workflow/`: workflow primitive composition (`Workflow` + step classes).
- `patterns/`: reusable orchestration patterns (`PlanExecutePattern`, `ProposeCriticPattern`, and related variants).
- `clients/`: LLM client setup plus representative `generate(LLMRequest(...))` calls.
- `model_selection/`: local-vs-remote backend policy decisions.
- `tools/`: unified tool runtime usage across callable/script/MCP sources.
- `optimization/`: optimization-oriented multi-step tool-routing flows.

All runnable examples:

- use public APIs,
- emit JSON outputs, and
- write trace metadata under `artifacts/examples/traces`.

## Quickstart

Run from repository root:

```bash
PYTHONPATH=src python examples/agents/direct_llm_call.py
```

Run the smoke test set:

```bash
make examples-smoke
```

## Deterministic Testing

Examples are capability-first and do not include deterministic branching logic.
Deterministic behavior is test-only and provided through `tests/example_monkeypatch/sitecustomize.py` when `DRA_EXAMPLE_LLM_MODE=deterministic` is set.
