"""Public-facing configuration for Agent Skills support."""

from __future__ import annotations

import os
from collections.abc import Sequence
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
        normalized_extra_paths = _normalize_path_sequence(
            raw_value=self.extra_paths,
            field_name="extra_paths",
        )
        normalized_pinned_skills = _normalize_string_sequence(
            raw_value=self.pinned_skills,
            field_name="pinned_skills",
        )
        if not isinstance(self.catalog_prompt_target, str):
            raise TypeError("catalog_prompt_target must be a string.")
        normalized_catalog_prompt_target = self.catalog_prompt_target.strip().lower()
        if normalized_catalog_prompt_target not in {"system", "user"}:
            raise ValueError("catalog_prompt_target must be either 'system' or 'user'.")
        if not isinstance(self.allow_automatic_activation, bool):
            raise TypeError("allow_automatic_activation must be a boolean.")

        object.__setattr__(self, "project_root", normalized_project_root)
        object.__setattr__(self, "extra_paths", normalized_extra_paths)
        object.__setattr__(self, "pinned_skills", normalized_pinned_skills)
        object.__setattr__(self, "catalog_prompt_target", normalized_catalog_prompt_target)


def _normalize_path_sequence(*, raw_value: object, field_name: str) -> tuple[str, ...]:
    """Normalize tuple-like path configuration while rejecting scalar strings."""
    if raw_value is None:
        return ()
    if isinstance(raw_value, (str, os.PathLike)):
        raise TypeError(f"{field_name} must be a sequence of paths, not a single string/path.")
    if not isinstance(raw_value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of paths.")
    return tuple(
        normalized_path
        for item in raw_value
        if (normalized_path := os.fspath(item).strip())
    )


def _normalize_string_sequence(*, raw_value: object, field_name: str) -> tuple[str, ...]:
    """Normalize tuple-like string configuration while rejecting scalar strings."""
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        raise TypeError(f"{field_name} must be a sequence of strings, not a single string.")
    if not isinstance(raw_value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    normalized_names: list[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} may only contain strings.")
        normalized_name = item.strip()
        if normalized_name:
            normalized_names.append(normalized_name)
    return tuple(normalized_names)
