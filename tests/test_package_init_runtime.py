from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
from pathlib import Path

import pytest


def test_package_init_falls_back_to_unknown_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_init = Path(__file__).resolve().parents[1] / "src" / "design_research_agents" / "__init__.py"
    spec = importlib.util.spec_from_file_location("dra_init_under_test", package_init)
    assert spec is not None
    assert spec.loader is not None

    def _missing_version(_dist_name: str) -> str:
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(importlib_metadata, "version", _missing_version)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.__version__ == "0+unknown"
    assert "__version__" in module.__dir__()
    assert "ModelSelector" in module.__dir__()
