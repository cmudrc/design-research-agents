"""Tool runtime adapter used by ``AgentRuntime`` agent-routing mode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec


class AgentRoutingToolRuntimeAdapter(ToolRuntime):
    """Expose named agent alternatives as router-selectable virtual tools."""

    def __init__(
        self,
        *,
        alternatives: Mapping[str, Agent],
        descriptions: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize adapter with named alternatives.

        Args:
            alternatives: Mapping from route/tool names to executable agents.
            descriptions: Optional human-readable descriptions by route name.
        """
        self._alternatives = {
            name.strip(): agent
            for name, agent in alternatives.items()
            if isinstance(name, str) and name.strip()
        }
        self._descriptions = {
            name.strip(): description.strip()
            for name, description in (descriptions or {}).items()
            if isinstance(name, str)
            and name.strip()
            and isinstance(description, str)
            and description.strip()
        }

    def list_tools(self) -> Sequence[ToolSpec]:
        """Return virtual tool specs derived from agent-routing alternatives.

        Returns:
            The resulting value.
        """
        specs: list[ToolSpec] = []
        for name, agent in self._alternatives.items():
            specs.append(
                ToolSpec(
                    name=name,
                    description=self._descriptions.get(
                        name,
                        f"Route request to agent alternative '{type(agent).__name__}'.",
                    ),
                    input_schema={
                        "type": "object",
                        "additionalProperties": True,
                    },
                    output_schema={
                        "type": "object",
                        "additionalProperties": True,
                    },
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
        """Record a routing choice as a successful virtual tool invocation.

        Args:
            tool_name: Parameter value.
            input_dict: Parameter value.
            request_id: Parameter value.
            dependencies: Parameter value.

        Returns:
            The resulting value.
        """
        if tool_name not in self._alternatives:
            return ToolResult(
                tool_name=tool_name,
                result={},
                ok=False,
                error=f"Unknown agent-routing alternative '{tool_name}'.",
                metadata={
                    "request_id": request_id,
                    "dependency_keys": sorted(dependencies.keys()),
                },
            )

        return ToolResult(
            tool_name=tool_name,
            result={
                "selected_alternative": tool_name,
                "tool_input": dict(input_dict),
            },
            ok=True,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "virtual": True,
            },
        )
