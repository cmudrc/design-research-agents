## Design-Focused Examples

This directory contains runnable engineering-design examples that map directly to library terminology.

All runnable examples:
- use public APIs,
- emit JSON output,
- include trace metadata under `artifacts/examples/traces`.

## Directory Guide

- `examples/agents`
  - Direct and multi-step agent entrypoints.
- `examples/workflow`
  - Workflow primitive composition (`Workflow` + step classes).
- `examples/patterns`
  - Reusable orchestration patterns (`PlanExecutePattern`, `ReflexionPattern`, etc.).
- `examples/clients`
  - LLM client configuration + representative `generate(LLMRequest(...))` calls.
- `examples/model_selection`
  - Local-vs-remote policy decisions.
- `examples/tools`
  - Tool runtime examples across core/script/MCP sources.
- `examples/optimization`
  - Optimization-oriented multi-step tool routing.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/agents/direct_llm_call.py
```

## Deterministic Testing Model

Examples are capability-first and do not include deterministic branching logic.
Deterministic behavior is test-only and provided through `tests/example_monkeypatch/sitecustomize.py` when `DRA_EXAMPLE_LLM_MODE=deterministic` is set.
