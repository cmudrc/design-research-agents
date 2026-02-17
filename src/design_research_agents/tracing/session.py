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
    """Field value for ``event_type``."""
    run_id: str
    """Field value for ``run_id``."""
    span_id: str
    """Field value for ``span_id``."""
    parent_span_id: str | None
    """Field value for ``parent_span_id``."""
    timestamp: str
    """Field value for ``timestamp``."""
    timestamp_ms: int
    """Field value for ``timestamp_ms``."""
    duration_ms: int | None = None
    """Field value for ``duration_ms``."""
    attributes: dict[str, object] = field(default_factory=dict)
    """Field value for ``attributes``."""
    event_index: int | None = None
    """Field value for ``event_index``."""

    def asdict(self) -> dict[str, object]:
        """Return JSON-serializable dictionary representation.

        Returns:
            Normalized trace event mapping.
        """
        payload = dataclasses.asdict(self)
        payload["attributes"] = _normalize_value(self.attributes)
        return payload


@dataclass(slots=True, frozen=True)
class _SpanInfo:
    """_SpanInfo class."""

    start_time: float
    """Field value for ``start_time``."""
    parent_span_id: str | None
    """Field value for ``parent_span_id``."""


class TraceSession:
    """Run-scoped trace session tracking open spans and sinks."""

    def __init__(self, *, run_id: str, sinks: list[TraceSink]) -> None:
        """Initialize a trace session with a run id and sinks.

        Args:
            run_id: Run identifier for this trace session.
            sinks: Trace sinks that will receive emitted events.
        """
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
        """Open a new span and emit its start event.

        Args:
            event_type: Event type label for the span start.
            parent_span_id: Optional parent span id.
            attributes: Event attributes payload.

        Returns:
            Generated span id.
        """
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
        """Finish a span and emit a completion event with duration.

        Args:
            event_type: Event type label for the span completion.
            span_id: Span identifier to close.
            attributes: Event attributes payload.
        """
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
        """Emit an event tied to an existing span.

        Args:
            event_type: Event type label for the span event.
            span_id: Span identifier for the event.
            attributes: Event attributes payload.
        """
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
        """Emit a standalone event payload to all sinks.

        Args:
            event_type: Event type label.
            span_id: Span identifier for the event.
            parent_span_id: Optional parent span id.
            attributes: Event attributes payload.
            duration_ms: Optional event duration in milliseconds.
        """
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
