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
        """Initialize an empty source map and resolved routing tables."""
        self._sources: dict[str, ToolSource] = {}
        self._routes: dict[str, _ToolRoute] = {}

    def add_source(self, source: ToolSource) -> None:
        """Add a source and rebuild routing tables."""
        self._sources[source.source_id] = source
        self._rebuild_routes()

    def remove_source(self, source_id: str) -> None:
        """Remove a source by id and rebuild routing tables."""
        self._sources.pop(source_id, None)
        self._rebuild_routes()

    def list_tools(self) -> Sequence[ToolSpec]:
        """List routed tool specs across all registered sources."""
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
        """Invoke a routed tool name."""
        self._rebuild_routes()
        route = self._routes.get(tool_name)

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
        if result.tool_name == tool_name:
            return result
        return ToolResult(
            tool_name=tool_name,
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

        self._routes = rebuilt_routes


__all__ = ["ToolRegistry"]
