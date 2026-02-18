# Python interpreter and tool commands used by all targets.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
SPHINX ?= $(PYTHON) -m sphinx

.PHONY: help \
	install install-dev install-all check-python \
	lint lint-fix fmt fmt-check type unit test qa qa-full coverage structure-check docstrings-check legacy-check baseline-integrity-check \
	examples-test examples-deterministic examples-metrics run-example \
	docs docs-build docs-open docs-linkcheck docs-check \
	ci pre-commit \
	clean distclean purge-ignored-junk \
	format format-check typecheck

help:
	@echo "Common targets:"
	@echo "  install           Install runtime dependencies (editable)."
	@echo "  install-dev       Install contributor tooling (editable)."
	@echo "  install-all       Install contributor + local backend extras."
	@echo "  qa                Fast local checks: lint, fmt-check, type, unit."
	@echo "  qa-full           Full gate: qa + structure/docstrings + coverage."
	@echo "  legacy-check      Fail on removed legacy/fallback/C901 patterns in src."
	@echo "  baseline-integrity-check  Validate baseline entries reference existing files."
	@echo "  docs-build        Build strict Sphinx HTML docs."
	@echo "  docs-open         Open docs index in your browser."
	@echo "  docs              Build docs and open index locally."
	@echo "  ci                CI parity checks."
	@echo "  clean             Remove generated artifacts only."

# Validate Python runtime matches project requirement.
check-python:
	@$(PYTHON) -c "import sys; import pathlib; print(f'Using Python {sys.version.split()[0]} at {pathlib.Path(sys.executable)}'); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" || (echo "Python >= 3.11 is required by pyproject.toml"; exit 1)

# Install editable package with runtime dependencies only.
install:
	$(PIP) install --upgrade pip
	$(PIP) install -e "."

# Install editable package with contributor tooling.
install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

# Install editable package with contributor tooling and optional local extras.
install-all:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,local]"

# Run lint checks without modifying files.
lint: check-python
	$(RUFF) check .

# Apply auto-fixable lint changes.
lint-fix: check-python
	$(RUFF) check . --fix

# Format source code in place.
fmt: check-python
	$(RUFF) format .

# Verify formatting without changing files.
fmt-check: check-python
	$(RUFF) format --check .

# Run static type checks.
type: check-python
	$(MYPY) src

# Run the unit test suite.
unit: check-python
	PYTHONPATH=src $(PYTEST)

# Fast local quality checks.
qa: lint fmt-check type unit

# Enforce structural module size thresholds.
structure-check: check-python
	$(PYTHON) scripts/check_structural_thresholds.py

# Enforce complete Google-style docstrings for src/examples/scripts.
docstrings-check: check-python
	@mkdir -p artifacts
	@if [ -n "$(DOCSTRING_CHANGED_FILES_FILE)" ]; then \
		CHANGED_FILES_FILE="$(DOCSTRING_CHANGED_FILES_FILE)"; \
	else \
		CHANGED_FILES_FILE="artifacts/docstrings_changed_files.txt"; \
		git diff --name-only --cached --diff-filter=ACMR > "$${CHANGED_FILES_FILE}"; \
	fi; \
	$(PYTHON) scripts/check_google_docstrings.py \
		--baseline scripts/google_docstrings_baseline.txt \
		--changed-files-file "$${CHANGED_FILES_FILE}"

# Fail when removed legacy/fallback paths or C901 suppressions reappear in src.
legacy-check: check-python
	$(PYTHON) scripts/check_no_legacy_paths.py

# Ensure baseline entries do not reference files that no longer exist.
baseline-integrity-check: check-python
	$(PYTHON) scripts/check_baseline_integrity.py

# Estimate line coverage for the stable unit-suite baseline.
coverage: check-python
	mkdir -p artifacts/coverage
	PYTHONPATH=src $(PYTEST) --ignore=tests/test_examples_non_streaming.py --ignore=tests/test_examples_streaming.py --ignore=tests/test_examples_script_shell.py --cov=src/design_research_agents --cov-report=term --cov-report=json:artifacts/coverage/coverage.json -q
	$(PYTHON) scripts/check_coverage_thresholds.py --coverage-json artifacts/coverage/coverage.json

# Full quality gate used by CI and release checks.
qa-full: qa structure-check docstrings-check coverage

# Run deterministic example tests and emit junit XML for metrics/badge generation.
examples-test: check-python
	mkdir -p artifacts/examples
	PYTHONPATH=src $(PYTEST) tests/test_examples_non_streaming.py tests/test_examples_streaming.py tests/test_examples_script_shell.py --junitxml=artifacts/examples/examples-deterministic.junit.xml -q

# Generate deterministic examples metrics and corresponding badges.
examples-metrics: check-python examples-test
	$(PYTHON) scripts/generate_examples_metrics.py
	$(PYTHON) scripts/generate_examples_badges.py

# Run a deterministic workflow example.
run-example: check-python
	PYTHONPATH=src $(PYTHON) examples/workflow/workflow_runtime.py

# Build strict Sphinx HTML documentation.
docs-build: check-python
	PYTHONPATH=src $(SPHINX) -b html docs docs/_build/html -n -W --keep-going -E

# Open built docs index in a local browser.
docs-open:
	@$(PYTHON) -c "import os,webbrowser; webbrowser.open('file://' + os.path.abspath('docs/_build/html/index.html'))"

# Build docs and open index locally.
docs: docs-build docs-open

# Run strict Sphinx link validation.
docs-linkcheck: check-python
	PYTHONPATH=src $(SPHINX) -b linkcheck docs docs/_build/linkcheck -W --keep-going -E

# Validate docs terminology and local path consistency.
docs-check: check-python
	$(PYTHON) scripts/check_docs_consistency.py

# Remove traces dirs, Sphinx build output, egg-info, and generated artifacts.
purge-ignored-junk:
	@echo "Removing traces/ directories, Sphinx _build, and egg-info..."
	@find . -type d -name traces -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf docs/_build
	@rm -rf src/design_research_agents.egg-info
	@rm -rf artifacts
	@find . -maxdepth 2 -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true

# Remove generated artifacts only (no source mutation).
clean: purge-ignored-junk

# Deeper cleanup alias for callers that expect distclean semantics.
distclean: clean

# Aggregate checks used by CI.
ci: qa-full legacy-check baseline-integrity-check docs-check

# Check set used by local pre-commit hook.
pre-commit: lint fmt-check type structure-check docstrings-check unit docs-build

# Compatibility aliases.
format: fmt
format-check: fmt-check
typecheck: type
test: unit
examples-deterministic: examples-test
