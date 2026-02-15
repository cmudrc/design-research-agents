"""Shared runtime controls for multi-mode agent execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True, frozen=True)
class RuntimeControls:
    """Execution limits and soft budget controls shared by runtime modes."""

    max_steps: int = 5
    max_iterations: int = 3
    max_tool_calls_per_step: int = 5
    execution_timeout_seconds_per_step: int = 5
    soft_max_latency_ms: int | None = None
    soft_max_usd: float | None = None
    streaming_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate control bounds."""
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if self.max_tool_calls_per_step < 1:
            raise ValueError("max_tool_calls_per_step must be >= 1.")
        if self.execution_timeout_seconds_per_step < 1:
            raise ValueError("execution_timeout_seconds_per_step must be >= 1.")
        if self.soft_max_latency_ms is not None and self.soft_max_latency_ms < 0:
            raise ValueError("soft_max_latency_ms must be >= 0 when provided.")
        if self.soft_max_usd is not None and self.soft_max_usd < 0:
            raise ValueError("soft_max_usd must be >= 0 when provided.")

    def asdict(self) -> dict[str, object]:
        """Return dictionary representation for metadata payloads."""
        return asdict(self)
