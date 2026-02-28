"""Tool specification payloads and runtime protocol contracts.

These definitions describe how tools are registered, invoked, and reported
across agents and runtimes in a provider-neutral manner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal, Protocol, Self


@dataclass(slots=True, frozen=True)
class ToolCostHints:
    """Approximate cost metadata associated with a tool invocation.

    Attributes:
        token_cost_estimate: Estimated token cost consumed by the tool.
        latency_ms_estimate: Estimated wall-clock latency in milliseconds.
        usd_cost_estimate: Estimated direct monetary cost in USD.
    """

    token_cost_estimate: int | None = None
    """Estimated token usage for one invocation."""
    latency_ms_estimate: int | None = None
    """Estimated end-to-end latency in milliseconds."""
    usd_cost_estimate: float | None = None
    """Estimated direct monetary cost in USD."""


@dataclass(slots=True, frozen=True)
class ToolSideEffects:
    """Declared side effects for one tool implementation."""

    filesystem_read: bool = False
    """Whether the tool reads from the filesystem."""
    filesystem_write: bool = False
    """Whether the tool writes to the filesystem."""
    network: bool = False
    """Whether the tool performs network I/O."""
    commands: tuple[str, ...] = ()
    """Command names the tool may execute, when command execution is used."""


@dataclass(slots=True, frozen=True)
class ToolMetadata:
    """Tool source and guardrail metadata surfaced to runtimes/agents."""

    source: Literal["core", "mcp", "script", "custom"] = "core"
    """Origin of the tool implementation."""
    side_effects: ToolSideEffects = field(default_factory=ToolSideEffects)
    """Declared operational side effects used for policy enforcement."""
    timeout_s: int = 30
    """Maximum allowed invocation time in seconds."""
    max_output_bytes: int = 65_536
    """Maximum serialized output size accepted by runtime wrappers."""
    risky: bool | None = None
    """Explicit risk marker; inferred from side effects when omitted."""
    server_id: str | None = None
    """Owning server id for remote tools (for example MCP), when applicable."""

    def __post_init__(self) -> None:
        """Infer ``risky`` from side effects when not explicitly provided."""
        # If the risk level is explicitly provided, use it as-is.
        if self.risky is not None:
            return

        # Otherwise, infer risk based on declared side effects.
        # This is a simple heuristic and can be adjusted as needed.
        is_risky = self.side_effects.filesystem_write or self.side_effects.network or bool(self.side_effects.commands)

        # Since the dataclass is frozen, we need to use object.__setattr__ to set the inferred risky value.
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
    """Stable tool identifier used for invocation."""
    description: str
    """Human-readable tool description used by planners and routers."""
    input_schema: dict[str, object]
    """JSON-schema-like input contract for tool calls."""
    output_schema: dict[str, object]
    """JSON-schema-like output contract for tool results."""
    metadata: ToolMetadata = field(default_factory=ToolMetadata)
    """Operational metadata and policy hints."""
    permissions: tuple[str, ...] = ()
    """Permission tags surfaced to callers and policy layers."""
    cost_hints: ToolCostHints = field(default_factory=ToolCostHints)
    """Optional cost estimates used by planning heuristics."""

    @property
    def json_schema(self) -> dict[str, object]:
        """Return the input schema for LLM tool-calling payloads.

        Returns:
            Input JSON schema mapping for this tool.
        """
        return self.input_schema


@dataclass(slots=True, frozen=True)
class ToolArtifact:
    """File-like artifact emitted by a tool invocation."""

    path: str
    """Filesystem path to the emitted artifact."""
    mime: str
    """MIME type describing artifact content."""


@dataclass(slots=True, frozen=True)
class ToolError:
    """Structured tool failure details."""

    type: str
    """Machine-readable error type identifier."""
    message: str
    """Human-readable error message."""


@dataclass(slots=True, frozen=True, init=False)
class ToolResult:
    """Result payload emitted from a tool runtime invocation."""

    tool_name: str
    """Name of the invoked tool."""
    ok: bool
    """True when invocation succeeded."""
    result: object
    """Primary tool return payload."""
    artifacts: tuple[ToolArtifact, ...]
    """Artifact list emitted by the invocation."""
    warnings: tuple[str, ...]
    """Non-fatal warnings produced during invocation."""
    error: ToolError | None
    """Structured error details when ``ok`` is false."""
    metadata: dict[str, object]
    """Supplemental runtime metadata for diagnostics and tracing."""

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
        """Initialize canonical tool result payload.

        Args:
            tool_name: Name of the invoked tool.
            ok: Invocation success flag.
            result: Primary result payload (defaults to empty mapping).
            artifacts: Raw or typed artifact entries to normalize.
            warnings: Warning messages to attach to the result.
            error: Error payload to normalize into ``ToolError``.
            metadata: Optional diagnostic metadata mapping.
        """
        # Normalize the result to an empty mapping if None is provided, ensuring consistent types for downstream
        # processing.
        resolved_result: object = result if result is not None else {}

        # Normalize artifacts into a consistent list of ToolArtifact instances, allowing for flexible input formats
        # while ensuring a standard output structure.
        resolved_artifacts: list[ToolArtifact] = []
        for artifact in artifacts:
            if isinstance(artifact, ToolArtifact):
                resolved_artifacts.append(artifact)
                continue
            # Mapping-based artifacts are normalized with conservative defaults so
            # policy/serialization layers can rely on required fields being present.
            path = str(artifact.get("path", ""))
            mime = str(artifact.get("mime", "application/octet-stream"))
            resolved_artifacts.append(ToolArtifact(path=path, mime=mime))

        # Normalize the error into a ToolError instance if it's provided in a compatible format, allowing for flexible
        # error reporting while ensuring a standard structure for error details.
        resolved_error: ToolError | None
        if isinstance(error, ToolError):
            resolved_error = error
        elif isinstance(error, Mapping):
            resolved_error = ToolError(
                type=str(error.get("type", "ToolError")),
                message=str(error.get("message", "Unknown tool error.")),
            )
        elif isinstance(error, str):
            # Plain-string errors are upgraded into structured ToolError payloads so
            # callers do not need to branch on error representation type.
            resolved_error = ToolError(type="ToolError", message=error)
        else:
            resolved_error = None

        # Since the dataclass is frozen, we need to use object.__setattr__ to set the fields after normalization.
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "ok", bool(ok))
        object.__setattr__(self, "result", resolved_result)
        object.__setattr__(self, "artifacts", tuple(resolved_artifacts))
        object.__setattr__(self, "warnings", tuple(str(item) for item in warnings))
        object.__setattr__(self, "error", resolved_error)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def result_dict(self) -> dict[str, object]:
        """Return the primary result payload normalized to a dictionary.

        Returns:
            Dictionary value when ``result`` is mapping-like, else ``{}``.
        """
        if isinstance(self.result, Mapping):
            return dict(self.result)
        return {}

    def result_list(self) -> list[object]:
        """Return the primary result payload normalized to a list.

        Returns:
            List value when ``result`` is a list/tuple, else ``[]``.
        """
        if isinstance(self.result, list):
            return list(self.result)
        if isinstance(self.result, tuple):
            return list(self.result)
        return []

    @property
    def error_message(self) -> str | None:
        """Return the normalized tool error message when present.

        Returns:
            Error message string, or ``None``.
        """
        if self.error is None:
            return None
        return self.error.message

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        """Return artifact paths in emitted order.

        Returns:
            Tuple of artifact path strings.
        """
        return tuple(artifact.path for artifact in self.artifacts)


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
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one tool using structured input and execution metadata payloads.

        Implementations should avoid raising for expected tool failures and
        instead return ``ToolResult(ok=False)`` with error details.

        Args:
            tool_name: Name of the tool to invoke.
            input: Tool input payload mapping.
            request_id: Request identifier for tracing.
            dependencies: Dependency payload mapping for the tool.

        Returns:
            Tool invocation result payload.
        """

    def close(self) -> None:
        """Release any runtime-owned resources.

        Implementations that do not own external resources may implement this
        as a no-op so callers can use a uniform lifecycle pattern.
        """
        return None

    def __enter__(self) -> Self:
        """Return this runtime for use in a ``with`` statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Close the runtime when exiting a ``with`` block."""
        del exc_type, exc, tb
        self.close()
        return None
