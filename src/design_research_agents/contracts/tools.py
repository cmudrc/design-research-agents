"""Tool specification payloads and runtime protocol contracts.

These definitions describe how tools are registered, invoked, and reported
across agents and runtimes in a provider-neutral manner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(slots=True, frozen=True)
class ToolCostHints:
    """Approximate cost metadata associated with a tool invocation.

    Attributes:
        token_cost_estimate: Estimated token cost consumed by the tool.
        latency_ms_estimate: Estimated wall-clock latency in milliseconds.
        usd_cost_estimate: Estimated direct monetary cost in USD.
    """

    token_cost_estimate: int | None = None
    latency_ms_estimate: int | None = None
    usd_cost_estimate: float | None = None


@dataclass(slots=True, frozen=True)
class ToolSideEffects:
    """Declared side effects for one tool implementation."""

    filesystem_read: bool = False
    filesystem_write: bool = False
    network: bool = False
    commands: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ToolMetadata:
    """Tool source and guardrail metadata surfaced to runtimes/agents."""

    source: Literal["core", "mcp", "script", "custom"] = "core"
    side_effects: ToolSideEffects = field(default_factory=ToolSideEffects)
    timeout_s: int = 30
    max_output_bytes: int = 65_536
    risky: bool | None = None
    server_id: str | None = None

    def __post_init__(self) -> None:
        """Infer ``risky`` from side effects when not explicitly provided."""
        if self.risky is not None:
            return
        is_risky = (
            self.side_effects.filesystem_write
            or self.side_effects.network
            or bool(self.side_effects.commands)
        )
        object.__setattr__(self, "risky", is_risky)


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Static description of a tool available to agent runtimes.

    Attributes:
        name: Stable tool identifier used for invocation.
        description: Natural-language description used for planning/routing.
        input_schema: JSON-schema-like object describing accepted inputs.
        output_schema: JSON-schema-like object describing tool outputs.
        metadata: Source/policy metadata for runtime enforcement.
        permissions: Optional permission labels associated with the tool.
        cost_hints: Optional cost estimates used for planning heuristics.
    """

    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    metadata: ToolMetadata = field(default_factory=ToolMetadata)
    permissions: tuple[str, ...] = ()
    cost_hints: ToolCostHints = field(default_factory=ToolCostHints)

    @property
    def json_schema(self) -> dict[str, object]:
        """Return the input schema for LLM tool-calling payloads."""
        return self.input_schema


@dataclass(slots=True, frozen=True)
class ToolArtifact:
    """File-like artifact emitted by a tool invocation."""

    path: str
    mime: str


@dataclass(slots=True, frozen=True)
class ToolError:
    """Structured tool failure details."""

    type: str
    message: str


@dataclass(slots=True, frozen=True, init=False)
class ToolResult:
    """Result payload emitted from a tool runtime invocation."""

    tool_name: str
    ok: bool
    result: object
    artifacts: tuple[ToolArtifact, ...]
    warnings: tuple[str, ...]
    error: ToolError | None
    metadata: dict[str, object]

    def __init__(
        self,
        *,
        tool_name: str,
        ok: bool,
        result: object | None = None,
        artifacts: Sequence[ToolArtifact | Mapping[str, object]] = (),
        warnings: Sequence[str] = (),
        error: ToolError | Mapping[str, object] | str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize canonical tool result payload."""
        resolved_result: object = result if result is not None else {}

        resolved_artifacts: list[ToolArtifact] = []
        for artifact in artifacts:
            if isinstance(artifact, ToolArtifact):
                resolved_artifacts.append(artifact)
                continue
            path = str(artifact.get("path", ""))
            mime = str(artifact.get("mime", "application/octet-stream"))
            resolved_artifacts.append(ToolArtifact(path=path, mime=mime))

        resolved_error: ToolError | None
        if isinstance(error, ToolError):
            resolved_error = error
        elif isinstance(error, Mapping):
            resolved_error = ToolError(
                type=str(error.get("type", "ToolError")),
                message=str(error.get("message", "Unknown tool error.")),
            )
        elif isinstance(error, str):
            resolved_error = ToolError(type="ToolError", message=error)
        else:
            resolved_error = None

        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "ok", bool(ok))
        object.__setattr__(self, "result", resolved_result)
        object.__setattr__(self, "artifacts", tuple(resolved_artifacts))
        object.__setattr__(self, "warnings", tuple(str(item) for item in warnings))
        object.__setattr__(self, "error", resolved_error)
        object.__setattr__(self, "metadata", dict(metadata or {}))


class ToolRuntime(Protocol):
    """Protocol for registering and invoking named tools.

    Implementations may be in-memory, remote, or hybrid, but must present the
    same listing and invocation interface to agents.
    """

    def list_tools(self) -> Sequence[ToolSpec]:
        """Return all currently registered tool specifications.

        Returned specs describe every tool callable through ``invoke``.

        Returns:
            Sequence of registered tool specifications.
        """

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one tool using structured input and execution metadata payloads.

        Implementations should avoid raising for expected tool failures and
        instead return ``ToolResult(ok=False)`` with error details.

        Args:
            tool_name: Name of the tool to invoke.
            input_dict: Tool input payload mapping.
            request_id: Request identifier for tracing.
            dependencies: Dependency payload mapping for the tool.

        Returns:
            Tool invocation result payload.
        """
