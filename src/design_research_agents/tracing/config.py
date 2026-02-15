"""Trace configuration helpers and sink construction."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .sinks import ConsoleTraceSink, JSONLTraceSink, TraceSink
from .utils import _sanitize_filename

UTC = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017


@dataclass(slots=True, frozen=True)
class TraceConfig:
    """Configuration for tracing output and sink behavior."""

    enabled: bool = True
    trace_dir: Path = Path("traces")
    enable_jsonl: bool = True
    enable_console: bool = True
    console_stream: TextIO = sys.stderr


_TRACE_CONFIG: TraceConfig | None = None


def configure_tracing(*, config: TraceConfig) -> None:
    """Set a global trace configuration overriding environment defaults."""
    global _TRACE_CONFIG
    _TRACE_CONFIG = config


def _resolve_trace_config() -> TraceConfig:
    if _TRACE_CONFIG is not None:
        return _TRACE_CONFIG
    return TraceConfig(
        enabled=_parse_bool_env("DRA_TRACE_ENABLED", True),
        trace_dir=Path(os.environ.get("DRA_TRACE_DIR", "traces")).expanduser(),
        enable_jsonl=_parse_bool_env("DRA_TRACE_JSONL", True),
        enable_console=_parse_bool_env("DRA_TRACE_CONSOLE", True),
        console_stream=sys.stderr,
    )


def _build_trace_path(config: TraceConfig, *, run_id: str) -> Path | None:
    if not config.enable_jsonl:
        return None
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_run_id = _sanitize_filename(run_id)
    return config.trace_dir / f"run_{timestamp}_{safe_run_id}.jsonl"


def _build_sinks(config: TraceConfig, *, trace_path: Path | None) -> list[TraceSink]:
    sinks: list[TraceSink] = []
    if config.enable_jsonl and trace_path is not None:
        sinks.append(JSONLTraceSink(trace_path))
    if config.enable_console:
        sinks.append(ConsoleTraceSink(config.console_stream))
    return sinks


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
