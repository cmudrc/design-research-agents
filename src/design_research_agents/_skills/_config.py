"""Public-facing configuration for Agent Skills support."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True, kw_only=True)
class SkillsConfig:
    """Immutable configuration for Agent Skills discovery and prompt wiring."""

    project_root: str | os.PathLike[str] = "."
    """Root used to resolve the default ``.agents/skills`` directory and relative extra paths."""

    extra_paths: tuple[str, ...] = ()
    """Additional skill-root directories searched after the project-local skills root."""

    pinned_skills: tuple[str, ...] = ()
    """Skill names preloaded into prompts for deterministic constructor-scoped behavior."""

    catalog_prompt_target: Literal["system", "user"] = "system"
    """Prompt location used for the discoverable skills catalog when automatic activation is enabled."""

    allow_automatic_activation: bool = True
    """Whether tool-capable flows should advertise and expose ``skills.activate``."""

    def __post_init__(self) -> None:
        """Normalize string/path inputs into stable immutable tuples."""
        normalized_project_root = os.fspath(self.project_root)
        normalized_extra_paths = tuple(os.fspath(path).strip() for path in self.extra_paths if os.fspath(path).strip())
        normalized_pinned_skills = tuple(name.strip() for name in self.pinned_skills if name.strip())

        object.__setattr__(self, "project_root", normalized_project_root)
        object.__setattr__(self, "extra_paths", normalized_extra_paths)
        object.__setattr__(self, "pinned_skills", normalized_pinned_skills)
