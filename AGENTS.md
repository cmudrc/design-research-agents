# AGENTS.md

## Purpose
This repository is the Python 3.12+ agent-execution layer in the
CMU Design Research Collective design-research ecosystem.
Favor small, composable changes that preserve deterministic behavior and runtime contracts.
Keep dependencies minimal and avoid unnecessary complexity.
Project maturity is pre-alpha; breaking changes are acceptable only with explicit user/requester permission.

## Setup
- Create and activate a virtual environment:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
- Reproducible release interpreter is set in `.python-version` (`3.12`).
- Install dependencies:
  - `make dev` for normal development (`editable + [dev]`, including release tooling)
- Use `PYTHONPATH=src` when running scripts/examples directly.

## Testing And Validation
Use the smallest useful check while iterating, then run full gates before merge.

- Fast local loop:
  - `make fmt`
  - `make lint`
  - `make type`
  - `PYTHONPATH=src pytest -q tests/<target>.py`
- If examples changed:
  - `make examples-smoke`
- If docs/readmes/public docs changed:
  - `make docs-check`
  - `make docs-build`
- Pre-merge baseline:
  - `make ci`
- Pre-publish baseline:
  - `make release-check`

## Public Vs Private Boundaries
- Compatibility guarantees are for curated top-level exports in `src/design_research_agents/__init__.py` and public facade modules:
  - Top-level includes selected core contracts (`ExecutionResult`, `LLMRequest`, `LLMMessage`, `LLMResponse`, `ToolResult`) in addition to entrypoint classes.
  - `design_research_agents.agent`
  - `design_research_agents.workflow`
  - `design_research_agents.patterns`
  - `design_research_agents.llm`
  - `design_research_agents.memory`
  - `design_research_agents.model_selection`
  - `design_research_agents.skills`
  - `design_research_agents.study`
  - `design_research_agents.tools`
- Underscored module paths are internal/unstable (for example `_implementations`, `_runtime`, `_contracts`, `_tracing`, `_schemas`, `_memory`).
- Prefer public imports in user-facing examples/docs. Use internal modules only when no public equivalent exists and the usage is intentional.
- Keep internal naming conventions intact: internal packages/modules should remain underscore-prefixed.

## Behavioral Guardrails
- Keep tests deterministic by default; avoid introducing network-dependent behavior into standard test paths.
- For intentionally breaking changes, get explicit user/requester approval first, then update tests/docs/examples in the same change.
- Preserve tool runtime safety defaults unless a change explicitly requires otherwise:
  - no network by default
  - writes constrained to `artifacts/` by default
  - command allowlist enforced for shell execution
- Do not reintroduce removed legacy/fallback paths; CI enforces this.
- Do not silence complexity issues with `# noqa: C901`; split code instead.

## Release Planning
- Do not create monthly milestone naming tables, themed release PR names, or
  calendar release branches as default maintenance.
- Prefer small issue/PR-scoped planning and package version releases driven by
  user-facing changes.
- Use GitHub milestones only for explicit, short-lived initiatives with an
  active owner; they are optional scheduling aids, not release gates.
- Name release branches and release PRs for the version or concrete change set
  they contain.
- When publishing, update package metadata, docs, examples, and GitHub
  Releases/PyPI notes as needed. Do not add README callouts that point to
  monthly milestones.

## Keep This File Up To Date
Update this file when contributor-facing workflow changes. In particular:

- If setup/check commands change, update this file with the new canonical commands.
- If public exports change, update all coupled artifacts in the same PR:
  - `src/design_research_agents/__init__.py`
  - `tests/test_public_api.py`
  - `docs/api.rst`
  - examples coverage for the new/removed export
- If examples move/rename, update docs/example links and run `make docs-check`.
- If new baseline-tracked docstring debt is introduced or files move, update baseline references and run `make baseline-integrity-check`.

## Scope
Keep guidance here lightweight and actionable. Put deep implementation details in code/docs, not in this file.
