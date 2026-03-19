"""Per-profile built-in engineering knowledge definitions."""

from __future__ import annotations

from ._aerospace import PROFILE as AEROSPACE_PROFILE
from ._mechanics import PROFILE as MECHANICS_PROFILE
from ._stem import PROFILE as STEM_PROFILE

BUILTIN_KNOWLEDGE_PROFILES = {
    profile.name: profile
    for profile in (
        AEROSPACE_PROFILE,
        MECHANICS_PROFILE,
        STEM_PROFILE,
    )
}

__all__ = [
    "AEROSPACE_PROFILE",
    "BUILTIN_KNOWLEDGE_PROFILES",
    "MECHANICS_PROFILE",
    "STEM_PROFILE",
]
