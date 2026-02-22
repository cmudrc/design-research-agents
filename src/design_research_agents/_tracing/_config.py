"""Trace tracer configuration and sink construction."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

from ._sinks import ConsoleTraceSink, JSONLTraceSink, TraceSink
from ._utils import _sanitize_filename

UTC = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017


@dataclass(slots=True, frozen=True)
class Tracer:
    """Explicitly configured tracer dependency injected into runtimes."""

    enabled: bool = True
    """Field value for ``enabled``."""
    trace_dir: Path = Path("traces")
    """Field value for ``trace_dir``."""
    enable_jsonl: bool = True
    """Field value for ``enable_jsonl``."""
    enable_console: bool = True
    """Field value for ``enable_console``."""
    console_stream: TextIO = sys.stderr
    """Field value for ``console_stream``."""

    def build_trace_path(self, *, run_id: str) -> Path | None:
        """Build a trace JSONL path for one run when JSONL sink is enabled.

        Args:
            run_id: Input value for this parameter.

        Returns:
            Computed return value.
        """
        if not self.enable_jsonl:
            return None
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_run_id = _sanitize_filename(run_id)
        return self.trace_dir / f"run_{timestamp}_{safe_run_id}.jsonl"

    def build_sinks(self, *, trace_path: Path | None) -> list[TraceSink]:
        """Build concrete sinks for this tracer configuration.

        Args:
            trace_path: Input value for this parameter.

        Returns:
            Computed return value.
        """
        sinks: list[TraceSink] = []
        if self.enable_jsonl and trace_path is not None:
            sinks.append(JSONLTraceSink(trace_path))
        if self.enable_console:
            sinks.append(ConsoleTraceSink(self.console_stream))
        return sinks

    def run_callable(
        self,
        *,
        agent_name: str,
        request_id: str,
        input_payload: Mapping[str, object],
        function: Callable[[], object],
        dependencies: Mapping[str, object] | None = None,
    ) -> object:
        """Run one callable wrapped in explicit trace session lifecycle.

        Args:
            agent_name: Agent name used in trace metadata.
            request_id: Request id used for trace run and file naming.
            input_payload: Input payload metadata for trace run start.
            function: Zero-argument callable to execute.
            dependencies: Optional dependency mapping for trace metadata.

        Returns:
            Function return value.
        """
        from ._context import finish_trace_run, start_trace_run

        scope = start_trace_run(
            agent_name=agent_name,
            request_id=request_id,
            input_payload=dict(input_payload),
            dependencies=dict(dependencies or {}),
            tracer=self,
        )
        try:
            value = function()
        except Exception as exc:
            finish_trace_run(scope, error=str(exc))
            raise
        finish_trace_run(scope, result=SimpleNamespace(success=True, output=value))
        return value

    def resolve_latest_trace_path(self, request_id: str) -> str | None:
        """Resolve latest emitted JSONL trace path for one request id.

        Args:
            request_id: Request identifier used in trace filenames.

        Returns:
            Latest matching trace file path, or ``None``.
        """
        if not self.enable_jsonl:
            return None
        if not self.trace_dir.exists():
            return None
        safe_request_id = _sanitize_filename(request_id)
        candidates = sorted(
            self.trace_dir.glob(f"run_*_{safe_request_id}.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        return str(candidates[0])

    def trace_info(self, request_id: str) -> dict[str, object]:
        """Return JSON-serializable trace metadata for one request id.

        Args:
            request_id: Request identifier associated with the trace run.

        Returns:
            Trace metadata payload.
        """
        return {
            "request_id": request_id,
            "trace_path": self.resolve_latest_trace_path(request_id),
            "trace_dir": str(self.trace_dir),
        }


__all__ = ["Tracer"]
