# Python interpreter and pip command used by all targets.
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install test coverage lint lint-fix format format-check typecheck run-example docs ci clean purge-ignored-junk

# Install a batteries-included development environment.
install:
	$(PIP) install --upgrade pip
	$(PIP) install -e "."

# Install a batteries-included development environment.
install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

# Install a batteries-included development environment.
install-all:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,local]"

# Run the unit test suite.
test:
	PYTHONPATH=src pytest

# Estimate line coverage for the stable unit-suite baseline.
coverage:
	mkdir -p artifacts/coverage
	PYTHONPATH=src pytest --ignore=tests/test_examples_non_streaming.py --ignore=tests/test_examples_streaming.py --cov=src/design_research_agents --cov-report=term --cov-report=json:artifacts/coverage/coverage.json -q

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

# Build Sphinx HTML documentation.
docs:
	sphinx-build -b html docs docs/_build/html

# Remove traces/ dirs anywhere + Sphinx build output + egg-info.
purge-ignored-junk:
	@echo "Removing traces/ directories, Sphinx _build, and egg-info..."
	@find . -type d -name traces -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf docs/_build
	@rm -rf src/design_research_agents.egg-info
	@rm -rf artifacts
	@find . -maxdepth 2 -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true

# Aggregate checks used by CI.
ci: lint format-check typecheck test

# Make me squeaky clean
clean: purge-ignored-junk lint-fix format

# Check for pre-commit
pre-commit: lint format-check typecheck test
