"""Unified runtime that merges tools from core, MCP, lazy, and custom sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec

from .config import ToolRuntimeConfig
from .core import CoreToolSource
from .policy import ToolPolicy, ToolPolicyConfig
from .registry import ToolRegistry
from .sources.inprocess_source import InProcessToolSource, ToolHandler


class UnifiedToolRuntime(ToolRuntime):
    """Tool runtime that routes calls across enabled tool sources."""

    def __init__(self, *, config: ToolRuntimeConfig | None = None) -> None:
        """Initialize runtime sources based on the provided configuration."""
        self._config = config or ToolRuntimeConfig()
        self._registry = ToolRegistry()

        core_policy = ToolPolicy(
            ToolPolicyConfig(
                workspace_root=self._config.core_tools.workspace_root,
                artifacts_dir=self._config.core_tools.artifacts_dir,
                allow_writes_outside_artifacts=self._config.core_tools.allow_writes_outside_artifacts,
                allow_network=self._config.core_tools.allow_network,
                allowed_commands=self._config.core_tools.allowed_commands,
            )
        )
        self._core_policy = core_policy

        self._core_source: CoreToolSource | None = None
        if self._config.core_tools.enabled:
            self._core_source = CoreToolSource(policy=self._core_policy)
            self._registry.add_source(self._core_source)

        self._custom_source = InProcessToolSource(source_id="custom")
        self._registry.add_source(self._custom_source)

        self._mcp_source = None
        if self._config.mcp.enabled and self._config.mcp.servers:
            from .sources.mcp_source import McpToolSource

            self._mcp_source = McpToolSource(
                mcp_config=self._config.mcp,
                policy=self._core_policy,
            )
            self._registry.add_source(self._mcp_source)

        self._lazy_source = None
        if self._config.lazy_tools.enabled:
            from .sources.lazy_source import LazyToolSource

            lazy_policy = ToolPolicy(
                ToolPolicyConfig(
                    workspace_root=self._config.core_tools.workspace_root,
                    artifacts_dir=self._config.core_tools.artifacts_dir,
                    allow_writes_outside_artifacts=(
                        self._config.lazy_tools.allow_writes_outside_artifacts
                    ),
                    allow_network=self._config.lazy_tools.allow_network,
                    allowed_commands=self._config.lazy_tools.allowed_commands,
                    default_timeout_s=self._config.lazy_tools.timeout_s_default,
                )
            )
            self._lazy_source = LazyToolSource(
                lazy_config=self._config.lazy_tools,
                policy=lazy_policy,
            )
            self._registry.add_source(self._lazy_source)

    @property
    def registry(self) -> ToolRegistry:
        """Return the source-merging registry."""
        return self._registry

    @property
    def config(self) -> ToolRuntimeConfig:
        """Return active runtime configuration."""
        return self._config

    def list_tools(self) -> Sequence[ToolSpec]:
        """List all tools currently exposed by enabled runtime sources."""
        return self._registry.list_tools()

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one tool through the registry routing layer."""
        return self._registry.invoke(
            tool_name,
            input_dict,
            request_id=request_id,
            dependencies=dependencies,
        )

    def register_tool(self, *, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a custom in-process tool."""
        self._custom_source.register_tool(spec=spec, handler=handler)

    def close(self) -> None:
        """Release external source resources."""
        if self._mcp_source is not None and hasattr(self._mcp_source, "close"):
            self._mcp_source.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Best-effort cleanup for external sources during GC."""
        self.close()


__all__ = ["UnifiedToolRuntime"]
