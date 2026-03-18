"""Internal dataclasses used by Agent Skills discovery and prompt injection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ._config import SkillsConfig


@dataclass(slots=True, frozen=True, kw_only=True)
class DiscoveredSkill:
    """Parsed immutable skill definition."""

    name: str
    description: str
    body: str
    skill_root: Path
    skill_file: Path
    compatibility: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    source_label: str

    def __post_init__(self) -> None:
        """Normalize mapping-like metadata into a plain dictionary."""
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(slots=True, frozen=True, kw_only=True)
class SkillCatalog:
    """Immutable ordered collection of discovered skills."""

    skills: tuple[DiscoveredSkill, ...]

    def names(self) -> tuple[str, ...]:
        """Return discovered skill names in deterministic order."""
        return tuple(skill.name for skill in self.skills)

    def by_name(self) -> dict[str, DiscoveredSkill]:
        """Return discovered skills keyed by name."""
        return {skill.name: skill for skill in self.skills}

    def get(self, skill_name: str) -> DiscoveredSkill | None:
        """Return one skill by name when present."""
        normalized_name = skill_name.strip()
        for skill in self.skills:
            if skill.name == normalized_name:
                return skill
        return None


@dataclass(slots=True, frozen=True, kw_only=True)
class SkillsContext:
    """Resolved constructor-time skills context."""

    config: SkillsConfig
    catalog: SkillCatalog
    pinned_skills: tuple[DiscoveredSkill, ...]

    @property
    def discovered_skill_names(self) -> tuple[str, ...]:
        """Return all discovered skill names."""
        return self.catalog.names()

    @property
    def pinned_skill_names(self) -> tuple[str, ...]:
        """Return pinned skill names in deterministic order."""
        return tuple(skill.name for skill in self.pinned_skills)


def dedupe_skill_names(names: Sequence[str]) -> tuple[str, ...]:
    """Return skill names deduplicated while preserving order."""
    ordered: dict[str, None] = {}
    for raw_name in names:
        normalized = raw_name.strip()
        if normalized:
            ordered.setdefault(normalized, None)
    return tuple(ordered)
