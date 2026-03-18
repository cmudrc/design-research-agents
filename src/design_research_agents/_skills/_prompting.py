"""Prompt and metadata helpers for Agent Skills support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._llm import LLMMessage
from design_research_agents._contracts._tools import ToolResult

from ._models import DiscoveredSkill, SkillsContext

_SKILLS_ACTIVATE_TOOL_NAME = "skills.activate"


def build_skills_metadata(
    *,
    skills_context: SkillsContext | None,
    tool_results: Sequence[ToolResult] = (),
) -> dict[str, object] | None:
    """Build canonical skills metadata for execution results."""
    if skills_context is None:
        return None

    activated_names: list[str] = list(skills_context.pinned_skill_names)
    for tool_result in tool_results:
        if tool_result.tool_name != _SKILLS_ACTIVATE_TOOL_NAME or not tool_result.ok:
            continue
        resolved_name = tool_result.result_dict().get("name")
        if isinstance(resolved_name, str) and resolved_name.strip():
            activated_names.append(resolved_name.strip())

    deduped_activated = _dedupe_names(activated_names)
    return {
        "discovered_skill_names": list(skills_context.discovered_skill_names),
        "pinned_skill_names": list(skills_context.pinned_skill_names),
        "activated_skill_names": list(deduped_activated),
    }


def merge_skills_metadata(
    *,
    metadata: Mapping[str, object],
    skills_context: SkillsContext | None,
    tool_results: Sequence[ToolResult] = (),
) -> dict[str, object]:
    """Merge canonical skills metadata into an existing metadata mapping."""
    merged = dict(metadata)
    skills_metadata = build_skills_metadata(skills_context=skills_context, tool_results=tool_results)
    if skills_metadata is not None:
        merged["skills"] = skills_metadata
    return merged


def inject_skills_into_prompt_pair(
    *,
    system_prompt: str,
    user_prompt: str,
    skills_context: SkillsContext | None,
    include_catalog: bool,
) -> tuple[str, str]:
    """Inject discoverable and pinned skills into a system/user prompt pair."""
    if skills_context is None:
        return system_prompt, user_prompt

    resolved_system_prompt = system_prompt
    resolved_user_prompt = user_prompt
    pinned_text = build_pinned_skills_text(skills_context.pinned_skills)
    if pinned_text:
        resolved_system_prompt = _append_section(
            prompt_text=resolved_system_prompt,
            section_label="Activated skills",
            section_body=pinned_text,
            fallback_prompt="Use the activated skills below when they are relevant.",
        )

    if include_catalog and skills_context.config.allow_automatic_activation and skills_context.discovered_skill_names:
        catalog_text = build_available_skills_text(skills_context)
        if skills_context.config.catalog_prompt_target == "system":
            resolved_system_prompt = _append_section(
                prompt_text=resolved_system_prompt,
                section_label="Available skills",
                section_body=catalog_text,
                fallback_prompt="Use the available skills when they are relevant to the task.",
            )
        else:
            resolved_user_prompt = _append_section(
                prompt_text=resolved_user_prompt,
                section_label="Available skills",
                section_body=catalog_text,
                fallback_prompt="Use the available skills when they are relevant to the task.",
            )

    return resolved_system_prompt, resolved_user_prompt


def inject_skills_into_messages(
    *,
    messages: Sequence[LLMMessage],
    skills_context: SkillsContext | None,
    include_catalog: bool,
) -> list[LLMMessage]:
    """Inject discoverable and pinned skills into normalized message lists."""
    if skills_context is None:
        return list(messages)

    existing_messages = list(messages)
    system_prompt = ""
    user_prompt = ""
    system_index: int | None = None
    user_index: int | None = None

    for index, message in enumerate(existing_messages):
        if message.role == "system" and system_index is None:
            system_index = index
            system_prompt = message.content
        if message.role == "user":
            user_index = index
            user_prompt = message.content

    injected_system_prompt, injected_user_prompt = inject_skills_into_prompt_pair(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        skills_context=skills_context,
        include_catalog=include_catalog,
    )

    if injected_system_prompt.strip():
        if system_index is None:
            existing_messages.insert(0, LLMMessage(role="system", content=injected_system_prompt))
        else:
            existing_messages[system_index] = LLMMessage(
                role="system",
                content=injected_system_prompt,
                name=existing_messages[system_index].name,
            )
    if injected_user_prompt.strip():
        if user_index is None:
            existing_messages.append(LLMMessage(role="user", content=injected_user_prompt))
        else:
            existing_messages[user_index] = LLMMessage(
                role="user",
                content=injected_user_prompt,
                name=existing_messages[user_index].name,
            )
    return existing_messages


def build_available_skills_text(skills_context: SkillsContext) -> str:
    """Return a compact discoverable-skills catalog for prompt injection."""
    skill_lines = [
        'When a skill is relevant, call `skills.activate` with {"skill_name": "<name>"} before relying on it.'
    ]
    for skill in skills_context.catalog.skills:
        skill_lines.extend(
            [
                f"- skill_name: {skill.name}",
                f"  description: {skill.description}",
            ]
        )
    return "\n".join(skill_lines)


def build_pinned_skills_text(skills: Sequence[DiscoveredSkill]) -> str:
    """Return preloaded skill instructions as a structured prompt block."""
    rendered: list[str] = []
    for skill in skills:
        compatibility = ", ".join(skill.compatibility) if skill.compatibility else "(none)"
        rendered.extend(
            [
                f'<active_skill name="{skill.name}" root="{skill.skill_root}">',
                f"description: {skill.description}",
                f"compatibility: {compatibility}",
                "instructions:",
                skill.body,
                "</active_skill>",
            ]
        )
    return "\n".join(rendered)


def _append_section(
    *,
    prompt_text: str,
    section_label: str,
    section_body: str,
    fallback_prompt: str,
) -> str:
    """Append a labeled section to prompt text while handling blank prompts."""
    normalized_body = section_body.strip()
    if not normalized_body:
        return prompt_text

    normalized_prompt = prompt_text.strip()
    parts = [normalized_prompt] if normalized_prompt else [fallback_prompt]
    parts.append(f"{section_label}:\n{normalized_body}")
    return "\n\n".join(parts)


def _dedupe_names(raw_names: Sequence[str]) -> tuple[str, ...]:
    """Return names deduplicated in insertion order."""
    ordered: dict[str, None] = {}
    for raw_name in raw_names:
        normalized_name = raw_name.strip()
        if normalized_name:
            ordered.setdefault(normalized_name, None)
    return tuple(ordered)
