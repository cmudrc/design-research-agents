# Python interpreter and pip command used by all targets.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy

.PHONY: install install-dev install-all check-python test coverage structure-check examples-deterministic examples-metrics lint lint-fix format format-check typecheck run-example docs ci clean purge-ignored-junk pre-commit

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

# Validate Python runtime matches project requirement.
check-python:
	@$(PYTHON) -c "import sys; import pathlib; print(f'Using Python {sys.version.split()[0]} at {pathlib.Path(sys.executable)}'); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" || (echo "Python >= 3.11 is required by pyproject.toml"; exit 1)

# Run the unit test suite.
test: check-python
	PYTHONPATH=src $(PYTEST)

# Estimate line coverage for the stable unit-suite baseline.
coverage: check-python
	mkdir -p artifacts/coverage
	PYTHONPATH=src $(PYTEST) --ignore=tests/test_examples_non_streaming.py --ignore=tests/test_examples_streaming.py --ignore=tests/test_examples_script_shell.py --cov=src/design_research_agents --cov-report=term --cov-report=json:artifacts/coverage/coverage.json -q
	$(PYTHON) scripts/check_coverage_thresholds.py --coverage-json artifacts/coverage/coverage.json

# Enforce structural module size thresholds.
structure-check: check-python
	$(PYTHON) scripts/check_structural_thresholds.py

# Run deterministic example tests and emit junit XML for metrics/badge generation.
examples-deterministic: check-python
	mkdir -p artifacts/examples
	PYTHONPATH=src $(PYTEST) tests/test_examples_non_streaming.py tests/test_examples_streaming.py tests/test_examples_script_shell.py --junitxml=artifacts/examples/examples-deterministic.junit.xml -q

# Generate deterministic examples metrics and corresponding badges.
examples-metrics: check-python examples-deterministic
	$(PYTHON) scripts/generate_examples_metrics.py
	$(PYTHON) scripts/generate_examples_badges.py

# Run lint checks without modifying files.
lint: check-python
	$(RUFF) check .

# Apply auto-fixable lint changes.
lint-fix: check-python
	$(RUFF) check . --fix

# Format source code in place.
format: check-python
	$(RUFF) format .

# Verify formatting without changing files.
format-check: check-python
	$(RUFF) format --check .

# Run static type checks.
typecheck: check-python
	$(MYPY) src

# Run a deterministic workflow example.
run-example: check-python
	PYTHONPATH=src $(PYTHON) examples/workflow/workflow_runtime.py

# Build Sphinx HTML documentation.
docs: check-python
	$(PYTHON) -m sphinx -b html docs docs/_build/html

# Remove traces/ dirs anywhere + Sphinx build output + egg-info.
purge-ignored-junk:
	@echo "Removing traces/ directories, Sphinx _build, and egg-info..."
	@find . -type d -name traces -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf docs/_build
	@rm -rf src/design_research_agents.egg-info
	@rm -rf artifacts
	@find . -maxdepth 2 -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true

# Aggregate checks used by CI.
ci: lint format-check typecheck structure-check test coverage

# Make me squeaky clean
clean: purge-ignored-junk lint-fix format

# Check for pre-commit
pre-commit: lint format-check typecheck structure-check test
