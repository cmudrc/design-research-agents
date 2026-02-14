# Python interpreter and pip command used by all targets.
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install test lint lint-fix format format-check typecheck run-example docs ci

# Install a batteries-included development environment.
install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,local]"

# Run the unit test suite.
test:
	pytest

# Run lint checks without modifying files.
lint:
	ruff check .

# Apply auto-fixable lint changes.
lint-fix:
	ruff check . --fix

# Format source code in place.
format:
	ruff format .

# Verify formatting without changing files.
format-check:
	ruff format --check .

# Run static type checks.
typecheck:
	mypy src

# Execute the router example script.
run-example:
	PYTHONPATH=src $(PYTHON) examples/router_agent.py

# Build Sphinx HTML documentation.
docs:
	sphinx-build -b html docs docs/_build/html

# Aggregate checks used by CI.
ci: lint format-check typecheck test
