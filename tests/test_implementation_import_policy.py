"""Policy test enforcing no legacy internal imports under implementations."""

from __future__ import annotations

from pathlib import Path


def test_implementations_forbid_legacy_internal_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    implementations_dir = root / "src" / "design_research_agents" / "implementations"
    banned = (
        "design_research_agents.agent.implementations",
        "design_research_agents.agent.internal",
        "design_research_agents.workflow.internal",
    )

    violations: list[str] = []
    for path in implementations_dir.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for needle in banned:
            if needle in content:
                violations.append(f"{path}: contains '{needle}'")

    assert violations == [], "\n".join(violations)
