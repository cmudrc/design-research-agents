PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
SPHINX ?= $(PYTHON) -m sphinx
BUILD ?= $(PYTHON) -m build
TWINE ?= $(PYTHON) -m twine
UV ?= uv
REPRO_PYTHON ?= $(shell cat .python-version 2>/dev/null || echo 3.12.12)
REPRO_EXTRAS ?= dev full

DOCSTRING_CHANGED_FILES_FILE ?=
DOCSTRING_CHANGED_FILES_DEFAULT := artifacts/docstrings_changed_files.txt

.PHONY: help check-python check-uv dev install-dev repro lock \
	lint fmt fmt-check type test qa ci coverage \
	release-check \
	structure-check docstrings-check legacy-check baseline-integrity-check junk-check \
	examples-smoke examples-test examples-metrics run-example \
	docs docs-build docs-check docs-linkcheck clean

help:
	@echo "Common targets:"
	@echo "  dev              Install project in editable mode with dev dependencies."
	@echo "  repro            Frozen reproducible install using uv.lock (default extras: dev full)."
	@echo "  lock             Regenerate uv.lock for release reproducibility."
	@echo "  install-dev      Alias for dev."
	@echo "  test             Run pytest suite."
	@echo "  qa               Run lint, fmt-check, type, and test."
	@echo "  ci               Full CI checks used in GitHub Actions."
	@echo "  coverage         Run tests with coverage threshold check."
	@echo "  release-check    Build sdist/wheel and run twine metadata checks."
	@echo "  docs             Build docs (same as docs-build)."
	@echo "  clean            Remove generated build artifacts and local caches."

check-python:
	@$(PYTHON) -c "import sys, pathlib; print(f'Using Python {sys.version.split()[0]} at {pathlib.Path(sys.executable)}'); raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" || (echo "Python >= 3.12 is required by pyproject.toml"; exit 1)

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv is required for lock/repro targets. Install it from https://docs.astral.sh/uv/getting-started/installation/"; exit 1)

dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-dev: dev

repro: check-uv
	$(UV) sync --frozen --python $(REPRO_PYTHON) $(foreach extra,$(REPRO_EXTRAS),--extra $(extra))

lock: check-uv
	$(UV) lock --python $(REPRO_PYTHON)

lint: check-python
	$(RUFF) check .

fmt: check-python
	$(RUFF) format .

fmt-check: check-python
	$(RUFF) format --check .

type: check-python
	$(MYPY) src

test: check-python
	PYTHONPATH=src $(PYTEST) -q

qa: lint fmt-check type test

structure-check: check-python
	$(PYTHON) scripts/check_structural_thresholds.py --repo-root .

docstrings-check: check-python
	@mkdir -p artifacts
	@CHANGED_FILES_FILE="$(DOCSTRING_CHANGED_FILES_FILE)"; \
	if [ -z "$${CHANGED_FILES_FILE}" ]; then \
		CHANGED_FILES_FILE="$(DOCSTRING_CHANGED_FILES_DEFAULT)"; \
		git diff --name-only --diff-filter=ACMR HEAD > "$${CHANGED_FILES_FILE}"; \
	fi; \
	$(PYTHON) scripts/check_google_docstrings.py \
		--baseline scripts/google_docstrings_baseline.txt \
		--changed-files-file "$${CHANGED_FILES_FILE}" \
		--enforce-codes DGS013,DGS014,DGS015

legacy-check: check-python
	$(PYTHON) scripts/check_no_legacy_paths.py

baseline-integrity-check: check-python
	$(PYTHON) scripts/check_baseline_integrity.py

junk-check: check-python
	$(PYTHON) scripts/check_no_tracked_junk.py

coverage: check-python
	mkdir -p artifacts/coverage
	PYTHONPATH=src $(PYTEST) --cov=src/design_research_agents --cov-report=term --cov-report=json:artifacts/coverage/coverage.json -q
	$(PYTHON) scripts/check_coverage_thresholds.py --coverage-json artifacts/coverage/coverage.json

release-check: check-python
	rm -rf dist
	$(BUILD)
	$(TWINE) check dist/*

examples-smoke: check-python
	PYTHONPATH=src $(PYTEST) -m examples_smoke -q

examples-test: check-python
	mkdir -p artifacts/examples
	PYTHONPATH=src $(PYTEST) -m examples_full --junitxml=artifacts/examples/examples-deterministic.junit.xml -q

examples-metrics: check-python examples-test
	$(PYTHON) scripts/generate_examples_metrics.py
	$(PYTHON) scripts/generate_examples_badges.py

run-example: check-python
	PYTHONPATH=src $(PYTHON) examples/workflow/workflow_runtime.py

docs-build: check-python
	$(PYTHON) scripts/generate_example_docs.py
	PYTHONPATH=src $(SPHINX) -b html docs docs/_build/html -n -W --keep-going -E

docs-check: check-python
	$(PYTHON) scripts/generate_example_docs.py --check
	$(PYTHON) scripts/check_docs_consistency.py

docs-linkcheck: check-python
	PYTHONPATH=src $(SPHINX) -b linkcheck docs docs/_build/linkcheck -W --keep-going -E

docs: docs-build

ci: qa coverage structure-check docstrings-check legacy-check baseline-integrity-check junk-check docs-check examples-smoke

clean:
	rm -rf docs/_build artifacts build htmlcov src/design_research_agents.egg-info .coverage coverage.xml
	find . -maxdepth 2 -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true
	find examples -type d -name artifacts -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -exec rm -f {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".coverage.*" \) -exec rm -f {} + 2>/dev/null || true
	find . -type d -name traces -prune -exec rm -rf {} + 2>/dev/null || true
