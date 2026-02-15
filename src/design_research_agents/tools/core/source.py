"""Core tool source that ships with the framework."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.tools import ToolResult, ToolSpec
from design_research_agents.tools.policy import ToolPolicy
from design_research_agents.tools.sources.inprocess_source import InProcessToolSource

from .bash_tools import register_bash_tools
from .data_tools import register_data_tools
from .fs_tools import register_fs_tools
from .git_tools import register_git_tools
from .math_tools import register_math_tools
from .run_tools import register_run_tools
from .search_tools import register_search_tools
from .text_tools import register_text_tools


class CoreToolSource:
    """In-process source exposing native framework tools."""

    source_id = "core"

    def __init__(self, *, policy: ToolPolicy) -> None:
        """Initialize core source and register default built-in tools."""
        self._policy = policy
        self._source = InProcessToolSource(source_id=self.source_id)
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        register_math_tools(self._source)
        register_text_tools(self._source)
        register_fs_tools(self._source, policy=self._policy)
        register_search_tools(self._source, policy=self._policy)
        register_git_tools(self._source, policy=self._policy)
        register_data_tools(self._source, policy=self._policy)
        register_run_tools(self._source, policy=self._policy)
        register_bash_tools(self._source)

    def list_tools(self) -> Sequence[ToolSpec]:
        """List all core tools after validating policy compatibility."""
        specs = tuple(self._source.list_tools())
        for spec in specs:
            self._policy.validate_tool_spec(spec)
        return specs

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one core tool with policy validation and artifact checks."""
        spec = next(
            (candidate for candidate in self._source.list_tools() if candidate.name == tool_name),
            None,
        )
        if spec is None:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"Tool '{tool_name}' is not registered.",
            )

        self._policy.validate_tool_spec(spec)
        result = self._source.invoke(
            tool_name,
            input_dict,
            request_id=request_id,
            dependencies=dependencies,
        )
        if not result.ok:
            return result
        self._policy.validate_result_artifacts(result)
        return result

    def register_tool(self, *, spec: ToolSpec, handler: object) -> None:
        """Compatibility hook for runtime shim usage."""
        # Compatibility method mirrors old BaseToolRuntime customization behavior.
        self._source.register_tool(spec=spec, handler=handler)  # type: ignore[arg-type]


__all__ = ["CoreToolSource"]
