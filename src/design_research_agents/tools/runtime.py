"""Unified runtime that merges tools from core, MCP, lazy, and custom sources."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec

from .config import (
    CoreToolsConfig,
    LazyToolsConfig,
    McpConfig,
    McpServerConfig,
    ToolRuntimeConfig,
    load_tool_runtime_config,
)
from .core import CoreToolSource
from .policy import ToolPolicy, ToolPolicyConfig
from .registry import ToolRegistry
from .sources.inprocess_source import InProcessToolSource, ToolHandler


class UnifiedToolRuntime(ToolRuntime):
    """Tool runtime that routes calls across enabled tool sources."""

    def __init__(
        self,
        *,
        workspace_root: str | os.PathLike[str] = ".",
        enable_core_tools: bool = True,
        lazy_search_paths: tuple[str | os.PathLike[str], ...] | None = None,
        mcp_servers: tuple[McpServerConfig, ...] | None = None,
    ) -> None:
        """Initialize runtime sources from ergonomic constructor arguments."""
        normalized_workspace_root = os.fspath(workspace_root)
        normalized_lazy_paths = (
            tuple(os.fspath(search_path) for search_path in lazy_search_paths)
            if lazy_search_paths is not None
            else None
        )

        lazy_tools_config = (
            LazyToolsConfig(
                enabled=bool(normalized_lazy_paths),
                search_paths=normalized_lazy_paths,
            )
            if normalized_lazy_paths is not None
            else LazyToolsConfig(enabled=False)
        )
        mcp_config = McpConfig(
            enabled=bool(mcp_servers),
            servers=tuple(mcp_servers or ()),
        )
        runtime_config = ToolRuntimeConfig(
            core_tools=CoreToolsConfig(
                enabled=enable_core_tools,
                workspace_root=normalized_workspace_root,
            ),
            mcp=mcp_config,
            lazy_tools=lazy_tools_config,
        )
        self._initialize_from_config(runtime_config)

    @classmethod
    def lazy(
        cls,
        *,
        search_paths: tuple[str | os.PathLike[str], ...],
        workspace_root: str | os.PathLike[str] = ".",
        enable_core_tools: bool = False,
    ) -> UnifiedToolRuntime:
        """Create a runtime focused on lazy tools."""
        return cls(
            workspace_root=workspace_root,
            enable_core_tools=enable_core_tools,
            lazy_search_paths=search_paths,
        )

    @classmethod
    def mcp(
        cls,
        *,
        servers: tuple[McpServerConfig, ...],
        workspace_root: str | os.PathLike[str] = ".",
        enable_core_tools: bool = False,
    ) -> UnifiedToolRuntime:
        """Create a runtime focused on MCP tools."""
        return cls(
            workspace_root=workspace_root,
            enable_core_tools=enable_core_tools,
            mcp_servers=servers,
        )

    @classmethod
    def from_yaml(cls, path: str) -> UnifiedToolRuntime:
        """Create a runtime from a YAML configuration file."""
        runtime_config = load_tool_runtime_config(path)
        instance = cls.__new__(cls)
        instance._initialize_from_config(runtime_config)
        return instance

    def _initialize_from_config(self, runtime_config: ToolRuntimeConfig) -> None:
        """Initialize runtime sources from a fully-resolved config object."""
        self._config = runtime_config
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
