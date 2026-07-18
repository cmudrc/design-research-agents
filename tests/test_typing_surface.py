"""Consumer-level checks for the advertised typed package surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from mypy import api as mypy_api


def test_lazy_top_level_exports_keep_concrete_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type-check representative imports as a downstream package would."""
    source = tmp_path / "consumer.py"
    source.write_text(
        """\
from collections.abc import Callable

from design_research_agents import MCPServerConfig, MultiStepAgent


def accepts_agent(agent: MultiStepAgent) -> None:
    pass


factory: Callable[..., MCPServerConfig] = MCPServerConfig.python_module
""",
        encoding="utf-8",
    )
    package_src = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("MYPYPATH", str(package_src))

    stdout, stderr, status = mypy_api.run(["--strict", "--no-incremental", str(source)])

    assert status == 0, stdout + stderr
