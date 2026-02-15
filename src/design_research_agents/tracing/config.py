"""Trace tracer configuration and sink construction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .sinks import ConsoleTraceSink, JSONLTraceSink, TraceSink
from .utils import _sanitize_filename

UTC = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017


@dataclass(slots=True, frozen=True)
class Tracer:
    """Explicitly configured tracer dependency injected into runtimes."""

    enabled: bool = True
    trace_dir: Path = Path("traces")
    enable_jsonl: bool = True
    enable_console: bool = True
    console_stream: TextIO = sys.stderr

    def build_trace_path(self, *, run_id: str) -> Path | None:
        """Build a trace JSONL path for one run when JSONL sink is enabled."""
        if not self.enable_jsonl:
            return None
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_run_id = _sanitize_filename(run_id)
        return self.trace_dir / f"run_{timestamp}_{safe_run_id}.jsonl"

    def build_sinks(self, *, trace_path: Path | None) -> list[TraceSink]:
        """Build concrete sinks for this tracer configuration."""
        sinks: list[TraceSink] = []
        if self.enable_jsonl and trace_path is not None:
            sinks.append(JSONLTraceSink(trace_path))
        if self.enable_console:
            sinks.append(ConsoleTraceSink(self.console_stream))
        return sinks


__all__ = ["Tracer"]
