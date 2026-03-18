"""Tool-runtime adapter exposing ``skills.activate`` for automatic skill activation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolResult,
    ToolRuntime,
    ToolSideEffects,
    ToolSpec,
)

from ._models import DiscoveredSkill, SkillsContext

_SKILLS_ACTIVATE_TOOL_NAME = "skills.activate"


class SkillsToolRuntimeAdapter:
    """Wrap an existing runtime and expose one skill-activation tool."""

    def __init__(self, *, wrapped_runtime: ToolRuntime, skills_context: SkillsContext) -> None:
        """Store dependencies and validate reserved tool-name collisions."""
        self._wrapped_runtime = wrapped_runtime
        self._skills_context = skills_context
        self._activation_spec = ToolSpec(
            name=_SKILLS_ACTIVATE_TOOL_NAME,
            description="Load the full instructions and resource paths for one discovered skill.",
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=ToolMetadata(
                source="custom",
                side_effects=ToolSideEffects(filesystem_read=True),
                timeout_s=10,
                risky=False,
            ),
        )
        self._ensure_reserved_tool_name_is_available()

    def list_tools(self) -> Sequence[ToolSpec]:
        """Return wrapped tools plus ``skills.activate``."""
        self._ensure_reserved_tool_name_is_available()
        return (*tuple(self._wrapped_runtime.list_tools()), self._activation_spec)

    def invoke(
        self,
        tool_name: str,
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke either the wrapped runtime or the skill activation tool."""
        if tool_name == _SKILLS_ACTIVATE_TOOL_NAME:
            return self._activate_skill(input=input)
        return self._wrapped_runtime.invoke(
            tool_name,
            input,
            request_id=request_id,
            dependencies=dependencies,
        )

    def close(self) -> None:
        """Release wrapped runtime resources."""
        self._wrapped_runtime.close()

    def __enter__(self) -> SkillsToolRuntimeAdapter:
        """Return this runtime adapter for use in ``with`` statements."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Close the wrapped runtime on context-manager exit."""
        del exc_type, exc, tb
        self.close()
        return None

    def _ensure_reserved_tool_name_is_available(self) -> None:
        """Fail fast when the wrapped runtime already exposes the reserved name."""
        wrapped_tool_names = {spec.name for spec in self._wrapped_runtime.list_tools()}
        if _SKILLS_ACTIVATE_TOOL_NAME in wrapped_tool_names:
            raise ValueError(f"ToolRuntime cannot expose reserved tool name '{_SKILLS_ACTIVATE_TOOL_NAME}'.")

    def _activate_skill(self, *, input: Mapping[str, object]) -> ToolResult:
        """Return the full body and resource inventory for one discovered skill."""
        raw_name = input.get("skill_name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            return ToolResult(
                tool_name=_SKILLS_ACTIVATE_TOOL_NAME,
                ok=False,
                error="skills.activate requires a non-empty string 'skill_name'.",
            )
        skill = self._skills_context.catalog.get(raw_name)
        if skill is None:
            return ToolResult(
                tool_name=_SKILLS_ACTIVATE_TOOL_NAME,
                ok=False,
                error=f"Unknown skill '{raw_name.strip()}'.",
            )

        return ToolResult(
            tool_name=_SKILLS_ACTIVATE_TOOL_NAME,
            ok=True,
            result={
                "name": skill.name,
                "description": skill.description,
                "compatibility": list(skill.compatibility),
                "skill_root": str(skill.skill_root),
                "instructions": skill.body,
                "resources": _build_resource_payload(skill),
                "metadata": dict(skill.metadata),
                "allowed_tools": list(skill.allowed_tools),
            },
        )


def _build_resource_payload(skill: DiscoveredSkill) -> dict[str, object]:
    """Return grouped absolute resource paths for standard skill subdirectories."""
    return {
        "references": _collect_files(skill.skill_root / "references"),
        "scripts": _collect_files(skill.skill_root / "scripts"),
        "assets": _collect_files(skill.skill_root / "assets"),
    }


def _collect_files(root: Path) -> list[str]:
    """Return absolute file paths under one optional resource root."""
    if not root.exists() or not root.is_dir():
        return []
    return [str(path.resolve()) for path in sorted(root.rglob("*")) if path.is_file()]
