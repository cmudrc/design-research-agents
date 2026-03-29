"""Parser for standard Agent Skills ``SKILL.md`` files."""

from __future__ import annotations

import re
from difflib import get_close_matches
from pathlib import Path

import yaml

from ._models import DiscoveredSkill

_FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_SUPPORTED_FRONTMATTER_KEYS = frozenset(
    {
        "allowed-tools",
        "compatibility",
        "description",
        "metadata",
        "name",
    }
)


def parse_skill_file(*, skill_file: Path, source_label: str) -> DiscoveredSkill:
    """Parse one ``SKILL.md`` file into an immutable skill definition."""
    raw_text = skill_file.read_text(encoding="utf-8")
    match = _FRONTMATTER_PATTERN.match(raw_text)
    if match is None:
        raise ValueError(f"{skill_file} must start with YAML frontmatter delimited by --- lines.")

    metadata_payload = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata_payload, dict):
        raise ValueError(f"{skill_file} frontmatter must decode to a mapping.")
    _validate_supported_frontmatter_keys(metadata_payload=metadata_payload, skill_file=skill_file)

    body = match.group(2).strip()
    if not body:
        raise ValueError(f"{skill_file} must contain a non-empty markdown body.")

    raw_name = metadata_payload.get("name")
    raw_description = metadata_payload.get("description")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"{skill_file} must declare a non-empty string 'name'.")
    if not isinstance(raw_description, str) or not raw_description.strip():
        raise ValueError(f"{skill_file} must declare a non-empty string 'description'.")

    name = raw_name.strip()
    if skill_file.parent.name != name:
        raise ValueError(f"{skill_file} name '{name}' must match parent directory '{skill_file.parent.name}'.")

    return DiscoveredSkill(
        name=name,
        description=raw_description.strip(),
        body=body,
        skill_root=skill_file.parent.resolve(),
        skill_file=skill_file.resolve(),
        compatibility=_normalize_string_sequence(metadata_payload.get("compatibility")),
        metadata=_normalize_metadata_mapping(metadata_payload.get("metadata")),
        allowed_tools=_normalize_string_sequence(metadata_payload.get("allowed-tools")),
        source_label=source_label,
    )


def _normalize_string_sequence(raw_value: object) -> tuple[str, ...]:
    """Normalize a string or list of strings into a tuple."""
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        return (normalized,) if normalized else ()
    if not isinstance(raw_value, list):
        raise ValueError("frontmatter sequence fields must be a string or list of strings.")

    normalized_values: dict[str, None] = {}
    for item in raw_value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("frontmatter sequence fields may only contain non-empty strings.")
        normalized_values.setdefault(item.strip(), None)
    return tuple(normalized_values)


def _normalize_metadata_mapping(raw_value: object) -> dict[str, str]:
    """Normalize optional metadata frontmatter into a ``dict[str, str]``."""
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("frontmatter 'metadata' must be a mapping of strings.")

    normalized: dict[str, str] = {}
    for raw_key, raw_item in raw_value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError("frontmatter metadata keys must be non-empty strings.")
        if not isinstance(raw_item, str):
            raise ValueError("frontmatter metadata values must be strings.")
        normalized[raw_key.strip()] = raw_item.strip()
    return normalized


def _validate_supported_frontmatter_keys(*, metadata_payload: dict[object, object], skill_file: Path) -> None:
    """Reject unknown frontmatter keys so skill definitions fail loudly."""
    unknown_keys = [key for key in metadata_payload if key not in _SUPPORTED_FRONTMATTER_KEYS]
    if not unknown_keys:
        return

    first_unknown = unknown_keys[0]
    if not isinstance(first_unknown, str):
        raise ValueError(f"{skill_file} frontmatter keys must be strings.")

    suggestion = get_close_matches(first_unknown, _SUPPORTED_FRONTMATTER_KEYS, n=1)
    suggestion_text = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    supported_keys = ", ".join(sorted(_SUPPORTED_FRONTMATTER_KEYS))
    raise ValueError(
        f"{skill_file} uses unsupported frontmatter key '{first_unknown}'.{suggestion_text} "
        f"Supported keys: {supported_keys}."
    )
