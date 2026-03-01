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
            Result produced by this call.
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
            tool_name: Value supplied for ``tool_name``.
            input_dict: Value supplied for ``input_dict``.
            request_id: Value supplied for ``request_id``.
            dependencies: Value supplied for ``dependencies``.

        Returns:
            Result produced by this call.
        """
