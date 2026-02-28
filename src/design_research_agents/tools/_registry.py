"""Unified registry that merges tools from multiple sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents._contracts._tools import ToolResult, ToolSpec
from design_research_agents._tracing import (
    emit_tool_invocation_observed,
    emit_tool_result_observed,
)

from ._sources._base import ToolSource


@dataclass(slots=True, frozen=True, kw_only=True)
class _ToolRoute:
    """Resolved registry entry for one externally visible tool name."""

    source_id: str
    """Identifier of the source that owns this tool."""
    source_tool_name: str
    """Underlying tool name used when invoking the source directly."""
    spec: ToolSpec
    """Normalized tool specification exposed by the registry."""


class ToolRegistry:
    """Merge and route tool calls across heterogeneous sources."""

    def __init__(self) -> None:
        """Initialize an empty source map and resolved routing tables."""
        self._sources: dict[str, ToolSource] = {}
        self._routes: dict[str, _ToolRoute] = {}

    def add_source(self, source: ToolSource) -> None:
        """Add a source and rebuild routing tables.

        Args:
            source: Tool source to register.
        """
        self._sources[source.source_id] = source
        self._rebuild_routes()

    def remove_source(self, source_id: str) -> None:
        """Remove a source by id and rebuild routing tables.

        Args:
            source_id: Identifier of the source to remove.
        """
        self._sources.pop(source_id, None)
        self._rebuild_routes()

    def list_tools(self) -> Sequence[ToolSpec]:
        """List routed tool specs across all registered sources.

        Returns:
            Stable snapshot of routed tool specifications.
        """
        # Recompute on demand so dynamically added/removed sources are immediately reflected.
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
        """Invoke a routed tool name.

        Args:
            tool_name: Public tool name requested by the caller.
            input_dict: Normalized tool-input payload.
            request_id: Trace-friendly request identifier for this invocation.
            dependencies: Runtime dependency bag forwarded to the selected source.

        Returns:
            Tool result normalized to the outward-facing tool name.
        """
        self._rebuild_routes()
        route = self._routes.get(tool_name)
        dependency_keys = sorted(str(key) for key in dependencies)

        if route is None:
            # Return a structured ToolResult for unknown names rather than raising so agent loops can recover.
            unknown_result = ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"Tool '{tool_name}' is not registered.",
            )
            emit_tool_result_observed(
                tool_name=tool_name,
                source_id=None,
                ok=False,
                result_payload=unknown_result,
                error=_resolve_error_message(unknown_result),
            )
            return unknown_result

        source = self._sources[route.source_id]
        emit_tool_invocation_observed(
            tool_name=tool_name,
            source_id=route.source_id,
            request_id=request_id,
            tool_input=input_dict,
            dependency_keys=dependency_keys,
        )
        try:
            result = source.invoke(
                route.source_tool_name,
                input_dict,
                request_id=request_id,
                dependencies=dependencies,
            )
        except Exception as exc:
            emit_tool_result_observed(
                tool_name=tool_name,
                source_id=route.source_id,
                ok=False,
                error=str(exc),
            )
            raise

        if result.tool_name == tool_name:
            normalized_result = result
        else:
            # Rebind the outward-facing tool name when a source uses a different internal alias.
            normalized_result = ToolResult(
                tool_name=tool_name,
                ok=result.ok,
                result=result.result,
                artifacts=result.artifacts,
                warnings=result.warnings,
                error=result.error,
                metadata=result.metadata,
            )
        emit_tool_result_observed(
            tool_name=tool_name,
            source_id=route.source_id,
            ok=normalized_result.ok,
            result_payload=normalized_result,
            error=_resolve_error_message(normalized_result),
        )
        return normalized_result

    def _rebuild_routes(self) -> None:
        """Recompute the outward-facing tool-name routing table.

        Raises:
            ValueError: If multiple sources expose the same public tool name.
        """
        rebuilt_routes: dict[str, _ToolRoute] = {}
        for source_id, source in sorted(self._sources.items()):
            for spec in source.list_tools():
                if spec.name in rebuilt_routes:
                    raise ValueError(f"Duplicate tool name '{spec.name}' from source '{source_id}'.")
                rebuilt_routes[spec.name] = _ToolRoute(
                    source_id=source_id,
                    source_tool_name=spec.name,
                    spec=spec,
                )

        self._routes = rebuilt_routes


def _resolve_error_message(result: ToolResult) -> str | None:
    """Return one normalized error message from a tool result payload."""
    if result.error is None:
        return None
    return result.error.message


__all__ = ["ToolRegistry"]
