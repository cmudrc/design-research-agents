"""Tests for deterministic Sphinx linkcheck result classification."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_linkcheck_results.py"


def _load_checker_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_linkcheck_results", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_results(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def _record(status: str, *, uri: str = "https://example.com") -> dict[str, object]:
    return {
        "filename": "docs/index.rst",
        "lineno": 12,
        "status": status,
        "code": 0,
        "uri": uri,
        "info": "test detail",
    }


def test_timeout_only_failure_is_non_blocking(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checker = _load_checker_module()
    results_path = tmp_path / "output.json"
    _write_results(results_path, _record("working"), _record("timeout"))

    exit_code = checker.main(["--results", str(results_path), "--sphinx-exit-code", "1"])

    assert exit_code == 0
    assert "Transient link timeouts (non-blocking)" in capsys.readouterr().out


def test_confirmed_broken_link_remains_blocking(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checker = _load_checker_module()
    results_path = tmp_path / "output.json"
    _write_results(results_path, _record("timeout"), _record("broken"))

    exit_code = checker.main(["--results", str(results_path), "--sphinx-exit-code", "1"])

    assert exit_code == 1
    assert "Confirmed broken links" in capsys.readouterr().out


def test_timeout_does_not_mask_fatal_sphinx_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checker = _load_checker_module()
    results_path = tmp_path / "output.json"
    _write_results(results_path, _record("timeout"))

    exit_code = checker.main(["--results", str(results_path), "--sphinx-exit-code", "2"])

    assert exit_code == 1
    assert "exit code 2" in capsys.readouterr().out


def test_unknown_status_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checker = _load_checker_module()
    results_path = tmp_path / "output.json"
    _write_results(results_path, _record("surprising-new-status"))

    exit_code = checker.main(["--results", str(results_path), "--sphinx-exit-code", "1"])

    assert exit_code == 1
    assert "unknown status" in capsys.readouterr().err
