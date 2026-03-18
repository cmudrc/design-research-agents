"""Discovery and validation for Agent Skills roots."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ._config import SkillsConfig
from ._models import DiscoveredSkill, SkillCatalog, SkillsContext, dedupe_skill_names
from ._parser import parse_skill_file

_DEFAULT_PROJECT_SKILLS_DIR = ".agents/skills"
_IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "build",
    "dist",
}
_MAX_DISCOVERY_DEPTH = 6


@dataclass(slots=True, frozen=True, kw_only=True)
class _SearchRoot:
    """Resolved search root with precedence metadata."""

    path: Path
    label: str
    required: bool


def resolve_skills_context(skills: SkillsConfig | None) -> SkillsContext | None:
    """Resolve constructor-scoped skills configuration into a discovered context."""
    if skills is None:
        return None

    catalog = discover_skills(skills)
    by_name = catalog.by_name()
    pinned_skills: list[DiscoveredSkill] = []
    for pinned_name in dedupe_skill_names(skills.pinned_skills):
        resolved = by_name.get(pinned_name)
        if resolved is None:
            raise ValueError(f"pinned skill '{pinned_name}' was not discovered.")
        pinned_skills.append(resolved)

    return SkillsContext(
        config=skills,
        catalog=catalog,
        pinned_skills=tuple(pinned_skills),
    )


def discover_skills(skills: SkillsConfig) -> SkillCatalog:
    """Discover skills under the configured project and extra roots."""
    project_root = Path(skills.project_root).expanduser().resolve()
    project_skills_root = project_root / _DEFAULT_PROJECT_SKILLS_DIR
    search_roots = [
        _SearchRoot(path=project_skills_root, label="project", required=False),
        *[
            _SearchRoot(
                path=_resolve_extra_path(extra_path=extra_path, project_root=project_root),
                label=f"extra:{index}",
                required=True,
            )
            for index, extra_path in enumerate(skills.extra_paths)
        ],
    ]

    discovered_by_name: dict[str, DiscoveredSkill] = {}
    for search_root in reversed(search_roots):
        if not search_root.path.exists():
            if search_root.required:
                raise ValueError(f"configured skills path does not exist: {search_root.path}")
            continue
        if not search_root.path.is_dir():
            raise ValueError(f"configured skills path is not a directory: {search_root.path}")

        for skill in _discover_skills_under_root(search_root):
            shadowed = discovered_by_name.get(skill.name)
            if shadowed is not None:
                warnings.warn(
                    (
                        f"Skill '{skill.name}' from {skill.source_label} shadows skill from "
                        f"{shadowed.source_label} at {shadowed.skill_file}."
                    ),
                    stacklevel=2,
                )
            discovered_by_name[skill.name] = skill

    ordered_skills = tuple(sorted(discovered_by_name.values(), key=lambda skill: skill.name))
    return SkillCatalog(skills=ordered_skills)


def _resolve_extra_path(*, extra_path: str, project_root: Path) -> Path:
    """Resolve one configured extra path relative to the project root when needed."""
    candidate = Path(extra_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _discover_skills_under_root(search_root: _SearchRoot) -> Iterable[DiscoveredSkill]:
    """Yield parsed skills under one resolved search root."""
    for directory in _walk_skill_directories(search_root.path):
        skill_file = directory / "SKILL.md"
        yield parse_skill_file(skill_file=skill_file, source_label=search_root.label)


def _walk_skill_directories(root: Path) -> Iterable[Path]:
    """Yield skill directories containing ``SKILL.md`` below ``root``."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current_path, depth = stack.pop()
        if depth > _MAX_DISCOVERY_DEPTH:
            continue
        try:
            entries = sorted(current_path.iterdir(), key=lambda path: path.name)
        except OSError:
            continue

        if (current_path / "SKILL.md").is_file():
            yield current_path
            continue

        for entry in reversed(entries):
            if not entry.is_dir():
                continue
            if entry.name in _IGNORED_DIR_NAMES:
                continue
            relative_parts = entry.relative_to(root).parts
            if relative_parts[:2] == ("docs", "_build"):
                continue
            stack.append((entry, depth + 1))
