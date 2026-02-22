#!/usr/bin/env python3
"""Generate an SVG coverage badge from pytest-cov JSON output."""

from __future__ import annotations

import json
from pathlib import Path

COVERAGE_JSON = Path("artifacts/coverage/coverage.json")
BADGE_SVG = Path(".github/badges/coverage.svg")


def _pick_color(percent: int) -> str:
    """Run pick color.

    Args:
        percent: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if percent >= 90:
        return "#4c1"  # brightgreen
    if percent >= 80:
        return "#97ca00"  # green
    if percent >= 70:
        return "#a4a61d"  # yellowgreen
    if percent >= 60:
        return "#dfb317"  # yellow
    if percent >= 50:
        return "#fe7d37"  # orange
    return "#e05d44"  # red


def _text_width(text: str) -> int:
    """Approximate text width in badge pixels.

    Args:
        text: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return 10 + (len(text) * 6)


def _render_badge(label: str, message: str, color: str) -> str:
    """Run render badge.

    Args:
        label: Input value for this parameter.
        message: Input value for this parameter.
        color: Input value for this parameter.

    Returns:
        Computed return value.
    """
    label_width = _text_width(label)
    message_width = _text_width(message)
    total_width = label_width + message_width
    label_x = label_width / 2
    message_x = label_width + (message_width / 2)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" 
    height="20" role="img" aria-label="{label}: {message}">
  <linearGradient id="g" x2="0" y2="100%">
    <stop offset="0" stop-color="#fff" stop-opacity=".7"/>
    <stop offset=".1" stop-color="#aaa" stop-opacity=".1"/>
    <stop offset=".9" stop-opacity=".3"/>
    <stop offset="1" stop-opacity=".5"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{message_width}" height="20" fill="{color}"/>
    <rect width="{total_width}" height="20" fill="url(#g)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" 
  font-size="11">
    <text x="{label_x:.1f}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_x:.1f}" y="14">{label}</text>
    <text x="{message_x:.1f}" y="15" fill="#010101" fill-opacity=".3">{message}</text>
    <text x="{message_x:.1f}" y="14">{message}</text>
  </g>
</svg>
"""


def _read_percent_display(path: Path) -> int:
    """Run read percent display.

    Args:
        path: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    totals = data.get("totals", {})
    raw_display = totals.get("percent_covered_display")
    if raw_display is None:
        raise ValueError("coverage JSON missing totals.percent_covered_display")

    normalized = str(raw_display).strip().rstrip("%")
    return int(float(normalized))


def main() -> None:
    """Generate coverage badge."""
    percent = _read_percent_display(COVERAGE_JSON)
    badge = _render_badge("Test Coverage", f"{percent}%", _pick_color(percent))
    BADGE_SVG.parent.mkdir(parents=True, exist_ok=True)
    BADGE_SVG.write_text(badge, encoding="utf-8")
    print(f"Wrote {BADGE_SVG} ({percent}%)")


if __name__ == "__main__":
    main()
