"""Trace sinks for JSONL and console output."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TextIO

from .utils import _preview


class TraceSink(Protocol):
    """Protocol for trace sinks that consume normalized event payloads."""

    def emit(self, event: dict[str, object]) -> None:
        """Handle one trace event payload.

        Args:
            event: Normalized trace event payload mapping.
        """

    def close(self) -> None:
        """Finalize and release sink resources."""


class JSONLTraceSink:
    """Append-only JSONL sink for trace events."""

    def __init__(self, path: Path) -> None:
        """Initialize a JSONL sink that appends to the given path.

        Args:
            path: Output path for the JSONL trace file.
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def emit(self, event: dict[str, object]) -> None:
        """Write one event to the JSONL file.

        Args:
            event: Normalized trace event payload mapping.
        """
        self._file.write(json.dumps(event, ensure_ascii=True))
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._file.close()


class ConsoleTraceSink:
    """Pretty streaming console sink for trace events."""

    def __init__(self, stream: TextIO | None = None) -> None:
        """Initialize a console sink that writes to the given stream.

        Args:
            stream: Optional stream to write console output to.
        """
        self._stream = stream if stream is not None else sys.stderr
        self._streaming_line_open = False

    def emit(self, event: dict[str, object]) -> None:
        """Render one event to the console stream.

        Args:
            event: Normalized trace event payload mapping.
        """
        event_type = str(event.get("event_type", ""))
        attrs = self._normalize_attributes(event.get("attributes", {}))

        if self._emit_token_delta(event_type=event_type, attrs=attrs):
            return
        self._maybe_close_streaming_line(event_type)

        rendered = self._render_event_line(event_type=event_type, attrs=attrs, event=event)
        if rendered is not None:
            self._write_line(rendered)

    def _normalize_attributes(self, attrs: object) -> Mapping[str, object]:
        """Run normalize attributes.

        Args:
            attrs: Parameter value.

        Returns:
            The resulting value.
        """
        if isinstance(attrs, Mapping):
            return attrs
        return {}

    def _emit_token_delta(self, *, event_type: str, attrs: Mapping[str, object]) -> bool:
        """Run emit token delta.

        Args:
            event_type: Parameter value.
            attrs: Parameter value.

        Returns:
            The resulting value.
        """
        if event_type != "ModelCallToken":
            return False
        delta_text = attrs.get("delta_text")
        if isinstance(delta_text, str) and delta_text:
            self._stream.write(delta_text)
            self._stream.flush()
            self._streaming_line_open = True
        return True

    def _maybe_close_streaming_line(self, event_type: str) -> None:
        """Run maybe close streaming line.

        Args:
            event_type: Parameter value.
        """
        if self._streaming_line_open and event_type.startswith("ModelCall"):
            self._stream.write("\n")
            self._streaming_line_open = False

    def _render_event_line(
        self,
        *,
        event_type: str,
        attrs: Mapping[str, object],
        event: Mapping[str, object],
    ) -> str | None:
        """Run render event line.

        Args:
            event_type: Parameter value.
            attrs: Parameter value.
            event: Parameter value.

        Returns:
            The resulting value.
        """
        event_lines: dict[str, str] = {
            "RunStarted": (
                f"[run] start run_id={event.get('run_id')} "
                f"agent={_preview(attrs.get('agent'))} "
                f"trace_path={_preview(attrs.get('trace_path'))}"
            ),
            "RunFinished": (
                f"[run] finished run_id={event.get('run_id')} "
                f"success={_preview(attrs.get('success'))} "
                f"duration_ms={_preview(event.get('duration_ms'))}"
            ),
            "AgentRunStarted": (
                f"[agent] start agent={_preview(attrs.get('agent'))} "
                f"request_id={_preview(attrs.get('request_id'))}"
            ),
            "AgentRunFinished": (
                f"[agent] finished agent={_preview(attrs.get('agent'))} "
                f"success={_preview(attrs.get('success'))} "
                f"duration_ms={_preview(event.get('duration_ms'))}"
            ),
            "ModelCallStarted": (
                f"[model] start model={_preview(attrs.get('model'))} "
                f"messages={_preview(attrs.get('message_count'))}"
            ),
            "ModelCallFinished": (
                f"[model] finished model={_preview(attrs.get('model'))} "
                f"duration_ms={_preview(event.get('duration_ms'))}"
            ),
            "ModelCallFailed": (
                f"[model] failed model={_preview(attrs.get('model'))} "
                f"error={_preview(attrs.get('error'))}"
            ),
            "ToolCallStarted": (
                f"[tool] start tool={_preview(attrs.get('tool_name'))} "
                f"input={_preview(attrs.get('tool_input'))}"
            ),
            "ToolCallFinished": (
                f"[tool] finished tool={_preview(attrs.get('tool_name'))} "
                f"duration_ms={_preview(event.get('duration_ms'))}"
            ),
            "ToolCallFailed": (
                f"[tool] failed tool={_preview(attrs.get('tool_name'))} "
                f"error={_preview(attrs.get('error'))}"
            ),
            "RouterDecision": (
                f"[router] decision tool={_preview(attrs.get('selected_tool_name'))} "
                f"source={_preview(attrs.get('source'))}"
            ),
            "ModelSelectionDecision": (
                f"[model-select] decision model={_preview(attrs.get('model_id'))} "
                f"provider={_preview(attrs.get('provider'))}"
            ),
            "ToolSelectionDecision": (
                f"[tool-select] decision tool={_preview(attrs.get('tool_name'))} "
                f"source={_preview(attrs.get('source'))}"
            ),
            "ContinuationDecision": (
                f"[continuation] step={_preview(attrs.get('step'))} "
                f"decision={_preview(attrs.get('continue'))} "
                f"source={_preview(attrs.get('source'))}"
            ),
            "GuardrailDecision": (
                f"[guardrail] guardrail={_preview(attrs.get('guardrail'))} "
                f"decision={_preview(attrs.get('decision'))} "
                f"reason={_preview(attrs.get('reason'))}"
            ),
        }
        return event_lines.get(event_type)

    def close(self) -> None:
        """Flush and close any open streaming output."""
        if self._streaming_line_open:
            self._stream.write("\n")
            self._stream.flush()
            self._streaming_line_open = False

    def _write_line(self, text: str) -> None:
        """Run write line.

        Args:
            text: Parameter value.
        """
        self._stream.write(f"{text}\n")
        self._stream.flush()
