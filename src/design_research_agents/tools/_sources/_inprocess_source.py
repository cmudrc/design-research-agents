"""In-process tool source for Python-callable tool handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from design_research_agents._contracts._tools import ToolResult, ToolSpec
from design_research_agents._tracing import (
    fail_tool_call,
    finish_tool_call,
    start_tool_call,
)

ToolHandler = Callable[[Mapping[str, object], str, Mapping[str, object]], object]


class InProcessToolSource:
    """Simple source that invokes Python handlers in-process."""

    def __init__(self, *, source_id: str) -> None:
        """Initialize in-process tool source storage for specs and handlers.

        Args:
            source_id: Input value for this parameter.
        """
        self.source_id = source_id
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register_tool(self, *, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register or replace one tool spec and its execution handler.

        Args:
            spec: Input value for this parameter.
            handler: Input value for this parameter.
        """
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def list_tools(self) -> Sequence[ToolSpec]:
        """List all currently registered in-process tools.

        Returns:
            Computed return value.
        """
        return tuple(self._specs.values())

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one in-process handler and normalize return payloads.

        Args:
            tool_name: Input value for this parameter.
            input_dict: Input value for this parameter.
            request_id: Input value for this parameter.
            dependencies: Input value for this parameter.

        Returns:
            Computed return value.
        """
        span_id = start_tool_call(
            tool_name=tool_name,
            tool_input=input_dict,
            request_id=request_id,
            dependencies=dependencies,
        )
        handler = self._handlers.get(tool_name)
        if handler is None:
            error_message = f"Tool '{tool_name}' is not registered."
            fail_tool_call(span_id, tool_name=tool_name, error=error_message)
            return ToolResult(tool_name=tool_name, ok=False, error=error_message)

        try:
            raw_result = handler(input_dict, request_id, dependencies)
        except Exception as exc:
            fail_tool_call(span_id, tool_name=tool_name, error=str(exc))
            return ToolResult(tool_name=tool_name, ok=False, error=str(exc))

        if isinstance(raw_result, ToolResult):
            finish_tool_call(span_id, tool_name=tool_name, result=raw_result)
            return raw_result

        if isinstance(raw_result, Mapping):
            result = ToolResult(
                tool_name=tool_name,
                ok=True,
                result=dict(raw_result),
                metadata={
                    "request_id": request_id,
                    "dependency_keys": sorted(dependencies.keys()),
                },
            )
            finish_tool_call(span_id, tool_name=tool_name, result=result)
            return result

        result = ToolResult(
            tool_name=tool_name,
            ok=True,
            result=raw_result,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
            },
        )
        finish_tool_call(span_id, tool_name=tool_name, result=result)
        return result


__all__ = ["InProcessToolSource", "ToolHandler"]
