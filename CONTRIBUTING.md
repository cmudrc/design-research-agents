# Contributing to design-research-agents

Thanks for helping improve this project.

This repository follows GitHub's recommended contributor-guideline pattern:
keep contribution expectations in a dedicated `CONTRIBUTING.md` so they are
easy to discover from issues, pull requests, and the repository homepage.

## Ways To Contribute

- Report bugs.
- Propose features or design improvements.
- Improve tests, docs, examples, and developer tooling.

## Before You Start

- Search existing issues and pull requests before opening a new one.
- For larger changes, open an issue first so we can align on scope.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

`.[dev]` is batteries-included and installs linting, formatting, typing, test,
docs, release, and pre-commit tooling.

## Local Quality Checks

Run these before opening a pull request:

```bash
make fmt
make lint
make type
make docstrings-check
make test
make docs
```

Optional but recommended:

```bash
pre-commit install
pre-commit run --all-files
```

## Pull Request Guidelines

- Keep PRs focused and reasonably small.
- Add or update tests for behavior changes.
- Update docs/examples when interfaces or workflows change.
- Link the issue in the PR description when applicable.
- Describe what changed, why, and how it was validated.

## Bug Reports

Please include:

- Clear expected vs actual behavior.
- Reproduction steps.
- Python version and OS.
- Relevant logs or tracebacks.

## Code Style

- Python 3.12+ target.
- Ruff for linting/formatting.
- Complete Google-style docstrings are required for all Python callables
  (including private/dunder) in `src/`, `examples/`, and `scripts/`.
  - Modules and classes must include a non-empty summary line.
  - Callables must include `Args`, `Returns`/`Yields`, and `Raises` sections
    whenever they apply.
- Mypy for type checking.
- Pytest for tests.

Minimal docstring templates:

```python
"""Module summary."""
```

```python
class Example:
    """Class summary."""
```

```python
def run(value: int) -> int:
    """Run the operation.

    Args:
        value: Input value.

    Returns:
        The computed value.

    Raises:
        ValueError: Raised when input is invalid.
    """
```

CI enforces these checks on Python 3.12.

## Conduct

Assume good intent, be respectful in technical discussions, and keep feedback
specific and actionable.
