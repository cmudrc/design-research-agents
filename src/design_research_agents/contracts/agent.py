"""Agent runtime protocol shared by all concrete agent implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from .execution import ExecutionResult


class Agent(Protocol):
    """Protocol that every agent implementation must satisfy.

    The protocol intentionally keeps the execution contract small: one
    non-streaming call with explicit runtime options and dependencies.
    """

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one agent run and return the final ``ExecutionResult`` payload.

        Implementations should treat ``prompt`` as the prompt text for one run.
        Use ``request_id`` and ``dependencies`` for run metadata and upstream
        dependency payloads.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final execution result payload.
        """
