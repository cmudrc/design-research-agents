# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
make dev
```

## Release Publishing

Before cutting a release, run `make release-check`. The GitHub `Publish`
workflow builds and validates distributions before any upload:

- Publishing a GitHub Release tagged `v{package-version}` publishes to PyPI.
- A manual workflow run is build-only by default.
- A recovery publish requires selecting the release tag and explicitly setting
  `publish=true`; publishing from a branch is rejected.
- Every publishing path rejects a tag that differs from the version in
  `pyproject.toml`.

## Local Quality Checks

Run these before opening a pull request:

```bash
make fmt
make lint
make type
make docstrings-check
make test
make docs-check
make examples-smoke
make ci
```

## Quality Gates

- `make coverage` enforces at least 95% total line coverage for the default deterministic suite.
- `make examples-test` executes the checked-in runnable examples.
- `make examples-coverage` requires every curated top-level `__all__` export to appear in at least one runnable example.

Optional but useful:

```bash
pre-commit install
pre-commit run --all-files
```

## Pull Request Guidelines

- Keep PRs focused and reasonably small.
- Add or update tests for behavior changes.
- Update docs/examples when interfaces or workflows change.
- Describe what changed and how you validated it.

## Code Style

- Python 3.12+ target
- Ruff for linting and formatting
- Mypy for type checking
- Pytest for tests
- Google-style docstrings in `src/`, `examples/`, and `scripts/`
