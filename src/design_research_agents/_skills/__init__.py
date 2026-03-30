"""Internal exports for Agent Skills discovery and prompt wiring."""

from ._config import SkillsConfig
from ._discovery import discover_skills, resolve_skills_context
from ._models import DiscoveredSkill, SkillCatalog, SkillsContext
from ._prompting import (
    build_available_skills_text,
    build_skills_metadata,
    inject_skills_into_messages,
    inject_skills_into_prompt_pair,
    merge_skills_metadata,
)
from ._runtime import SkillsToolRuntimeAdapter

__all__ = [
    "DiscoveredSkill",
    "SkillCatalog",
    "SkillsConfig",
    "SkillsContext",
    "SkillsToolRuntimeAdapter",
    "build_available_skills_text",
    "build_skills_metadata",
    "discover_skills",
    "inject_skills_into_messages",
    "inject_skills_into_prompt_pair",
    "merge_skills_metadata",
    "resolve_skills_context",
]
