"""Enforce complete public-API representation in runnable examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_METRICS = Path("artifacts/examples/examples_metrics.json")


def main() -> int:
    """Read generated metrics and enforce the configured API-in-examples floor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--minimum", type=float, default=100.0)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    public_api = metrics["public_api"]
    covered = int(public_api["covered_exports"])
    total = int(public_api["total_exports"])
    percentage = float(public_api["coverage_percent"])

    print(f"API in examples: {percentage:.1f}% ({covered}/{total})")
    for symbol in public_api.get("missing_symbols", []):
        print(f"- {symbol}")
    if percentage < args.minimum:
        print(f"Coverage threshold failed: {percentage:.1f}% < {args.minimum:.1f}%")
        return 1
    print("API-in-examples threshold passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
