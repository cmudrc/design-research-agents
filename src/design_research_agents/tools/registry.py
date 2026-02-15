"""Unified registry that merges tools from multiple sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.contracts.tools import ToolResult, ToolSpec

from .sources.base import ToolSource


@dataclass(slots=True, frozen=True)
class _ToolRoute:
    source_id: str
    source_tool_name: str
    spec: ToolSpec


class ToolRegistry:
    """Merge and route tool calls across heterogeneous sources."""

    def __init__(self) -> None:
        self._sources: dict[str, ToolSource] = {}
        self._routes: dict[str, _ToolRoute] = {}
        self._alias_map: dict[str, str] = {}

    def add_source(self, source: ToolSource) -> None:
        self._sources[source.source_id] = source
        self._rebuild_routes()

    def remove_source(self, source_id: str) -> None:
        self._sources.pop(source_id, None)
        self._rebuild_routes()

    def list_tools(self) -> Sequence[ToolSpec]:
        self._rebuild_routes()
        return tuple(route.spec for route in self._routes.values())

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        self._rebuild_routes()
        canonical_name = tool_name
        route = self._routes.get(canonical_name)
        if route is None:
            aliased = self._alias_map.get(tool_name)
            route = self._routes.get(aliased) if aliased is not None else None
            canonical_name = aliased or tool_name

        if route is None:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"Tool '{tool_name}' is not registered.",
            )

        source = self._sources[route.source_id]
        result = source.invoke(
            route.source_tool_name,
            input_dict,
            request_id=request_id,
            dependencies=dependencies,
        )
        if result.tool_name == canonical_name:
            return result
        return ToolResult(
            tool_name=canonical_name,
            ok=result.ok,
            result=result.result,
            artifacts=result.artifacts,
            warnings=result.warnings,
            error=result.error,
            metadata=result.metadata,
        )

    def _rebuild_routes(self) -> None:
        rebuilt_routes: dict[str, _ToolRoute] = {}
        for source_id, source in sorted(self._sources.items()):
            for spec in source.list_tools():
                if spec.name in rebuilt_routes:
                    raise ValueError(
                        f"Duplicate tool name '{spec.name}' from source '{source_id}'."
                    )
                rebuilt_routes[spec.name] = _ToolRoute(
                    source_id=source_id,
                    source_tool_name=spec.name,
                    spec=spec,
                )

        alias_counts: dict[str, int] = {}
        for tool_name in rebuilt_routes:
            if "::" not in tool_name:
                continue
            alias = tool_name.split("::", 1)[1]
            alias_counts[alias] = alias_counts.get(alias, 0) + 1

        alias_map: dict[str, str] = {}
        for tool_name in rebuilt_routes:
            if "::" not in tool_name:
                continue
            alias = tool_name.split("::", 1)[1]
            if alias_counts.get(alias, 0) != 1:
                continue
            if alias in rebuilt_routes:
                continue
            alias_map[alias] = tool_name

        self._routes = rebuilt_routes
        self._alias_map = alias_map


__all__ = ["ToolRegistry"]
