"""Source interfaces used by the unified tool registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from design_research_agents._contracts._tools import ToolResult, ToolSpec


class ToolSource(Protocol):
    """Pluggable source that can list and invoke tools."""

    source_id: str

    def list_tools(self) -> Sequence[ToolSpec]:
        """Return all tools exposed by the source.

        Returns:
            Computed return value.
        """

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke a tool by source-native name.

        Args:
            tool_name: Input value for this parameter.
            input_dict: Input value for this parameter.
            request_id: Input value for this parameter.
            dependencies: Input value for this parameter.

        Returns:
            Computed return value.
        """
