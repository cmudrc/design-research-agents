"""Policy checks for internal import boundaries and naming conventions."""

from __future__ import annotations

from pathlib import Path


def test_internal_packages_forbid_legacy_internal_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    implementations_dir = root / "src" / "design_research_agents" / "_implementations"
    banned = (
        "design_research_agents.agent.implementations",
        "design_research_agents.agent.internal",
        "design_research_agents.workflow.internal",
        "agent.implementations",
        "agent.internal",
        "workflow.internal",
    )

    violations: list[str] = []
    for path in implementations_dir.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for needle in banned:
            if needle in content:
                violations.append(f"{path}: contains '{needle}'")

    assert violations == [], "\n".join(violations)


def test_internal_modules_and_packages_use_underscore_naming() -> None:
    root = Path(__file__).resolve().parents[1]
    package_root = root / "src" / "design_research_agents"
    public_package_paths = {
        ".",
        "agent",
        "patterns",
        "workflow",
        "llm",
        "llm/clients",
        "memory",
        "skills",
        "tools",
    }
    public_module_paths = {
        "__init__.py",
        "workflow/workflow.py",
    }

    violations: list[str] = []
    for path in package_root.rglob("*"):
        relative = path.relative_to(package_root).as_posix()
        if "__pycache__" in path.parts:
            continue

        if path.is_dir():
            if not (path / "__init__.py").exists():
                continue
            if relative in public_package_paths:
                continue
            if not path.name.startswith("_"):
                violations.append(f"{path}: internal package name must start with '_'")
            continue

        if path.suffix != ".py":
            continue
        if path.name in {"__init__.py", "__main__.py"}:
            continue
        if relative in public_module_paths:
            continue
        if not path.name.startswith("_"):
            violations.append(f"{path}: internal module filename must start with '_'")

    assert violations == [], "\n".join(violations)
