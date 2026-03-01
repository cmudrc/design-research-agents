#!/usr/bin/env python3
"""Generate examples status and API-coverage SVG badges."""

from __future__ import annotations

import json
from pathlib import Path

METRICS_JSON = Path("artifacts/examples/examples_metrics.json")
EXAMPLES_BADGE_SVG = Path(".github/badges/examples-passing.svg")
API_COVERAGE_BADGE_SVG = Path(".github/badges/examples-api-coverage.svg")


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


def _read_metrics(path: Path) -> tuple[int, int, float, int, int]:
    """Run read metrics.

    Args:
        path: Input value for this parameter.

    Returns:
        Computed return value.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = data.get("examples", {})
    public_api = data.get("public_api", {})

    passed = int(examples.get("passed", 0))
    total = int(examples.get("total", 0))
    pass_percent = float(examples.get("pass_percent", 0.0))
    covered_exports = int(public_api.get("covered_exports", 0))
    total_exports = int(public_api.get("total_exports", 0))
    return passed, total, pass_percent, covered_exports, total_exports


def _format_percent(percent: float) -> str:
    """Run format percent.

    Args:
        percent: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return str(int(percent))


def main() -> None:
    """Generate and write SVG badges for deterministic example metrics."""
    passed, total, pass_percent, covered_exports, total_exports = _read_metrics(METRICS_JSON)
    api_coverage_percent = round((covered_exports / total_exports) * 100, 1) if total_exports else 100.0

    examples_badge = _render_badge(
        "Examples Passing",
        f"{passed}/{total}",
        _pick_color(round(pass_percent)),
    )
    api_badge = _render_badge(
        "Example API Coverage",
        f"{covered_exports}/{total_exports}",
        _pick_color(round(api_coverage_percent)),
    )

    EXAMPLES_BADGE_SVG.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_BADGE_SVG.write_text(examples_badge, encoding="utf-8")
    API_COVERAGE_BADGE_SVG.write_text(api_badge, encoding="utf-8")
    print(
        "Wrote "
        f"{EXAMPLES_BADGE_SVG} and {API_COVERAGE_BADGE_SVG} "
        f"(examples: {passed}/{total}, api: {covered_exports}/{total_exports})"
    )


if __name__ == "__main__":
    main()
