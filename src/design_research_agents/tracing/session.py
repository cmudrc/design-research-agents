"""Trace event models and span session management."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from .sinks import TraceSink
from .utils import _normalize_value

UTC = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017


@dataclass(slots=True, frozen=True)
class TraceEvent:
    """Normalized trace event payload."""

    event_type: str
    run_id: str
    span_id: str
    parent_span_id: str | None
    timestamp: str
    timestamp_ms: int
    duration_ms: int | None = None
    attributes: dict[str, object] = field(default_factory=dict)
    event_index: int | None = None

    def asdict(self) -> dict[str, object]:
        """Return JSON-serializable dictionary representation."""
        payload = dataclasses.asdict(self)
        payload["attributes"] = _normalize_value(self.attributes)
        return payload


@dataclass(slots=True, frozen=True)
class _SpanInfo:
    start_time: float
    parent_span_id: str | None


class TraceSession:
    """Run-scoped trace session tracking open spans and sinks."""

    def __init__(self, *, run_id: str, sinks: list[TraceSink]) -> None:
        """Initialize a trace session with a run id and sinks."""
        self.run_id = run_id
        self.root_span_id = uuid4().hex
        self._sinks = sinks
        self._open_spans: dict[str, _SpanInfo] = {}
        self._event_index = 0

    def start_span(
        self,
        event_type: str,
        *,
        parent_span_id: str | None,
        attributes: dict[str, object],
    ) -> str:
        """Open a new span and emit its start event."""
        span_id = uuid4().hex
        self._open_spans[span_id] = _SpanInfo(
            start_time=time.perf_counter(),
            parent_span_id=parent_span_id,
        )
        self.emit_event(
            event_type,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )
        return span_id

    def finish_span(
        self,
        event_type: str,
        *,
        span_id: str,
        attributes: dict[str, object],
    ) -> None:
        """Finish a span and emit a completion event with duration."""
        info = self._open_spans.pop(span_id, None)
        duration_ms = None
        parent_span_id = None
        if info is not None:
            duration_ms = int((time.perf_counter() - info.start_time) * 1000)
            parent_span_id = info.parent_span_id
        self.emit_event(
            event_type,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attributes=attributes,
            duration_ms=duration_ms,
        )

    def emit_span_event(
        self,
        event_type: str,
        *,
        span_id: str,
        attributes: dict[str, object],
    ) -> None:
        """Emit an event tied to an existing span."""
        parent_span_id = None
        info = self._open_spans.get(span_id)
        if info is not None:
            parent_span_id = info.parent_span_id
        self.emit_event(
            event_type,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )

    def emit_event(
        self,
        event_type: str,
        *,
        span_id: str,
        parent_span_id: str | None,
        attributes: dict[str, object],
        duration_ms: int | None = None,
    ) -> None:
        """Emit a standalone event payload to all sinks."""
        timestamp = datetime.now(UTC).isoformat()
        event = TraceEvent(
            event_type=event_type,
            run_id=self.run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            timestamp=timestamp,
            timestamp_ms=int(time.time() * 1000),
            duration_ms=duration_ms,
            attributes=dict(attributes),
            event_index=self._event_index,
        )
        payload = event.asdict()
        self._event_index += 1
        for sink in self._sinks:
            sink.emit(payload)

    def close(self) -> None:
        """Close all sinks associated with this session."""
        for sink in self._sinks:
            sink.close()
