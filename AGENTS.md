# AGENTS.md

## Purpose
This repository is a Python 3.12+ framework for engineering design agent research and experimentation.
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

## Release Naming
- Theme: famous folks in STEM, with a bias toward women and historically underrepresented figures when possible.
- Monthly release names are shared across milestone titles, release PR titles, and release branches.
  - Milestone title / PR title: `{base name} - {Month YYYY}`
  - Release branch: slugified full title, for example `lovelace-lift-may-2026`
- Milestone descriptions must use:
  - `Tracks {previous month YYYY} work.`
  - `Theme source: <url>`
- Release PR bodies must repeat the same `Theme source:` link used on the milestone.
- Never reuse an exact base name or the same primary subject across any month or any of the four design-research module repos unless all four `AGENTS.md` files are intentionally updated together.
- Before adding a new release name, check the `Release Naming` tables in all four repos to avoid repeats.
- `The April Alignment` is a carry-forward exception for the April 1, 2026 milestone and remains unchanged.

| Due date | Base name | Source subject |
| --- | --- | --- |
| April 1, 2026 | The April Alignment | Carry-forward exception (existing April 2026 milestone) |
| May 1, 2026 | Lovelace Lift | Ada Lovelace |
| June 1, 2026 | Mayer Momentum | Maria Goeppert Mayer |
| July 1, 2026 | Johnson Jumpstart | Katherine Johnson |
| August 1, 2026 | Jemison Journey | Mae Jemison |
| September 1, 2026 | Noether Nexus | Emmy Noether |
| October 1, 2026 | Ochoa Orbit | Ellen Ochoa |
| November 1, 2026 | Ride Relay | Sally Ride |
| December 1, 2026 | Doudna Drive | Jennifer Doudna |
| January 1, 2027 | Jackson Junction | Shirley Ann Jackson |

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
