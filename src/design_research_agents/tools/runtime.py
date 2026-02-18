"""Toolbox runtime that merges core, MCP, script, and in-process tools."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from design_research_agents.contracts.tools import (
    ToolMetadata,
    ToolResult,
    ToolRuntime,
    ToolSpec,
)

from .config import (
    CallableTool,
    CoreToolsConfig,
    McpConfig,
    McpServer,
    ScriptTool,
    ScriptToolsConfig,
    ToolRuntimeConfig,
)
from .core import CoreToolSource
from .policy import ToolPolicy, ToolPolicyConfig
from .registry import ToolRegistry
from .sources.inprocess_source import InProcessToolSource, ToolHandler


class Toolbox(ToolRuntime):
    """Tool runtime that routes calls across enabled tool sources."""

    def __init__(
        self,
        *,
        workspace_root: str | os.PathLike[str] = ".",
        enable_core_tools: bool = True,
        script_tools: tuple[ScriptTool, ...] | None = None,
        callable_tools: tuple[CallableTool, ...] | None = None,
        mcp_servers: tuple[McpServer, ...] | None = None,
    ) -> None:
        """Initialize toolbox sources from ergonomic constructor arguments.

        Args:
            workspace_root: Root directory for tools that interact with the filesystem.
            enable_core_tools: Whether to enable the built-in core tools.
            script_tools: Optional tuple of ScriptTool definitions to expose through a script tool
                source.
            callable_tools: Optional tuple of CallableTool definitions to register as in-process
                tools.
            mcp_servers: Optional tuple of MCP server definitions to connect to and expose tools
                from.
        """
        normalized_workspace_root = os.fspath(workspace_root)

        runtime_config = ToolRuntimeConfig(
            core_tools=CoreToolsConfig(
                enabled=enable_core_tools,
                workspace_root=normalized_workspace_root,
            ),
            mcp=McpConfig(
                enabled=bool(mcp_servers),
                servers=tuple(mcp_servers or ()),
            ),
            script_tools=ScriptToolsConfig(
                enabled=bool(script_tools),
                tools=tuple(script_tools or ()),
            ),
        )
        self._initialize_from_config(runtime_config)

        for callable_tool in tuple(callable_tools or ()):  # register after custom source exists.
            self.register_callable_tool(callable_tool)

    def _initialize_from_config(self, runtime_config: ToolRuntimeConfig) -> None:
        """Initialize runtime sources from a fully-resolved config object.

        Args:
            runtime_config: Fully-resolved runtime configuration object.

        Returns:
            None
        """
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

        self._script_source = None
        if self._config.script_tools.enabled and self._config.script_tools.tools:
            from .sources.script_source import ScriptToolSource

            script_policy = ToolPolicy(
                ToolPolicyConfig(
                    workspace_root=self._config.core_tools.workspace_root,
                    artifacts_dir=self._config.core_tools.artifacts_dir,
                    allow_writes_outside_artifacts=self._config.core_tools.allow_writes_outside_artifacts,
                    allow_network=self._config.core_tools.allow_network,
                    allowed_commands=self._config.core_tools.allowed_commands,
                    default_timeout_s=30,
                )
            )
            self._script_source = ScriptToolSource(
                script_tools=self._config.script_tools.tools,
                policy=script_policy,
            )
            self._registry.add_source(self._script_source)

    @property
    def registry(self) -> ToolRegistry:
        """Return the source-merging registry.

        Returns:
            The resulting value.
        """
        return self._registry

    @property
    def config(self) -> ToolRuntimeConfig:
        """Return active runtime configuration.

        Returns:
            The resulting value.
        """
        return self._config

    def list_tools(self) -> Sequence[ToolSpec]:
        """List all tools currently exposed by enabled runtime sources.

        Returns:
            Sequence of ToolSpec objects representing all tools currently exposed by enabled runtime
            sources, in no particular order.
        """
        return self._registry.list_tools()

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one tool through the registry routing layer.

        Args:
            tool_name: Name of the tool to invoke. This will be normalized by stripping leading and
                trailing whitespace before lookup.
            input_dict: Mapping of input values to provide for this tool invocation. This will be
                validated against the tool's input schema before invocation.
            request_id: Request ID to associate with this tool invocation, which will be passed
                through to the underlying tool handler and can be used for logging, tracing, and
                other purposes.
            dependencies: Mapping of dependencies to provide for this tool invocation, which will
                be passed through to the underlying tool handler and can be used to provide
                additional context or resources needed for the tool execution.

        Returns:
            The result of the tool invocation, as returned by the underlying tool handler. This will
            be validated against the tool's output schema before being returned to the caller.
        """
        return self._registry.invoke(
            tool_name,
            input_dict,
            request_id=request_id,
            dependencies=dependencies,
        )

    def register_tool(self, *, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a custom in-process tool.

        Args:
            spec: ToolSpec defining the tool to register. The name field will be normalized by
                stripping leading and trailing whitespace, and must be non-empty after
                normalization.
            handler: ToolHandler function to execute when this tool is invoked. The handler will be
                wrapped to match the expected signature for in-process tools, which includes
                additional parameters for request ID and dependencies that will be ignored by the
                provided handler.

        Returns:
            None
        """
        self._custom_source.register_tool(spec=spec, handler=handler)

    def register_callable_tool(self, callable_tool: CallableTool) -> None:
        """Register one callable tool wrapper.

        Args:
            callable_tool: CallableTool definition to register. The name field will be normalized by
                stripping leading and trailing whitespace, and must be non-empty after
                normalization.

        Returns:
            None

        Raises:
            Exception: Raised when execution fails.
        """
        normalized_name = callable_tool.name.strip()
        if not normalized_name:
            raise ValueError("CallableTool.name must be non-empty.")

        spec = ToolSpec(
            name=normalized_name,
            description=callable_tool.description,
            input_schema=dict(callable_tool.input_schema),
            output_schema=dict(callable_tool.output_schema),
            permissions=callable_tool.permissions,
            metadata=ToolMetadata(source="custom", risky=callable_tool.risky),
        )

        def _handler(
            input_dict: Mapping[str, object],
            _request_id: str,
            _dependencies: Mapping[str, object],
        ) -> object:
            """Run handler.

            Args:
                input_dict: Parameter value.
                _request_id: Parameter value.
                _dependencies: Parameter value.

            Returns:
                The resulting value.
            """
            del _request_id, _dependencies
            return callable_tool.handler(input_dict)

        self._custom_source.register_tool(spec=spec, handler=_handler)

    def close(self) -> None:
        """Release external source resources."""
        if self._mcp_source is not None and hasattr(self._mcp_source, "close"):
            self._mcp_source.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Best-effort cleanup for external sources during GC."""
        self.close()


__all__ = ["Toolbox"]
