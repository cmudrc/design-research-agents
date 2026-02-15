"""Tool source for manifest-less lazy scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.tools import (
    ToolMetadata,
    ToolResult,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools.config import LazyToolsConfig
from design_research_agents.tools.lazy.discovery import (
    LazyDiscoveryDiagnostic,
    LazyToolDefinition,
    discover_lazy_tools,
)
from design_research_agents.tools.lazy.runner import run_lazy_tool
from design_research_agents.tools.policy import ToolPolicy


class LazyToolSource:
    """Discover and execute local docblock-defined lazy tools."""

    source_id = "lazy"

    def __init__(self, *, lazy_config: LazyToolsConfig, policy: ToolPolicy) -> None:
        self._config = lazy_config
        self._policy = policy
        self._definitions: dict[str, LazyToolDefinition] = {}
        self._diagnostics: list[LazyDiscoveryDiagnostic] = []

    @property
    def diagnostics(self) -> Sequence[LazyDiscoveryDiagnostic]:
        """Return discovery diagnostics for invalid lazy scripts."""
        return tuple(self._diagnostics)

    def list_tools(self) -> Sequence[ToolSpec]:
        self._refresh_discovery()
        specs: list[ToolSpec] = []
        for canonical_name, definition in sorted(self._definitions.items()):
            header = definition.header
            input_properties: dict[str, object] = {}
            required_inputs: list[str] = []
            for input_spec in header.inputs:
                input_properties[input_spec.name] = _input_type_to_schema(input_spec.input_type)
                if input_spec.default is None:
                    required_inputs.append(input_spec.name)

            specs.append(
                ToolSpec(
                    name=canonical_name,
                    description=header.description,
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "properties": input_properties,
                        "required": required_inputs,
                    },
                    output_schema={"type": "object"},
                    metadata=ToolMetadata(
                        source="lazy",
                        side_effects=ToolSideEffects(
                            filesystem_read=header.capabilities.filesystem_read,
                            filesystem_write=header.capabilities.filesystem_write,
                            network=header.capabilities.network,
                            commands=header.capabilities.commands,
                        ),
                        timeout_s=header.timeout_s or self._config.timeout_s_default,
                        max_output_bytes=self._policy.config.default_max_output_bytes,
                        risky=True,
                    ),
                )
            )

        return tuple(specs)

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        self._refresh_discovery()
        definition = self._definitions.get(tool_name)
        if definition is None:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"Unknown lazy tool '{tool_name}'.",
            )

        return run_lazy_tool(
            tool_name=tool_name,
            definition=definition,
            input_dict=input_dict,
            policy=self._policy,
        )

    def _refresh_discovery(self) -> None:
        tools, diagnostics = discover_lazy_tools(self._config.search_paths)
        definitions: dict[str, LazyToolDefinition] = {}
        for definition in tools:
            canonical_name = f"lazy::{definition.header.tool_name}"
            definitions[canonical_name] = definition
        self._definitions = definitions
        self._diagnostics = diagnostics


def _input_type_to_schema(input_type: str) -> dict[str, object]:
    if input_type in {"str", "path"}:
        return {"type": "string"}
    if input_type == "int":
        return {"type": "integer"}
    if input_type == "float":
        return {"type": "number"}
    if input_type == "bool":
        return {"type": "boolean"}
    if input_type == "json":
        return {"type": "object"}
    if input_type == "list[str]":
        return {"type": "array", "items": {"type": "string"}}
    if input_type == "list[int]":
        return {"type": "array", "items": {"type": "integer"}}
    if input_type.startswith("enum[") and input_type.endswith("]"):
        values = [item.strip() for item in input_type[5:-1].split(",") if item.strip()]
        return {"type": "string", "enum": values}
    return {"type": "string"}


__all__ = ["LazyToolSource"]
