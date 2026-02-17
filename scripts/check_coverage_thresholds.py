"""Validate coverage percentages against release-blocking thresholds."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# Thresholds are pinned to the current stable baseline to keep CI deterministic
# during pre-alpha refactors.
GLOBAL_THRESHOLD = 83.0
PACKAGE_THRESHOLDS = {
    "workflow": 86.0,
    "agent": 76.0,
    "tools": 85.0,
    "llm": 74.0,
}


@dataclass(slots=True, frozen=True)
class CoverageSummary:
    """Minimal line-coverage summary for one scope."""

    covered_lines: int
    """Number of executable lines covered by tests."""
    num_statements: int
    """Total number of executable statements measured."""

    @property
    def percent(self) -> float:
        """Return line coverage percentage in [0, 100].

        Returns:
            Coverage percentage for this scope.
        """
        if self.num_statements <= 0:
            return 0.0
        return (self.covered_lines / self.num_statements) * 100.0


def _read_coverage(path: Path) -> dict[str, object]:
    """Read and validate a pytest-cov JSON payload.

    Args:
        path: Path to the coverage JSON report.

    Returns:
        Parsed JSON payload.

    Raises:
        ValueError: If the payload root is not a JSON object.
    """
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Coverage JSON payload must be an object.")
    return loaded


def _extract_global_summary(payload: dict[str, object]) -> CoverageSummary:
    """Extract global coverage totals from a coverage JSON payload.

    Args:
        payload: Parsed coverage JSON payload.

    Returns:
        Global coverage summary.

    Raises:
        ValueError: If required global totals are missing.
    """
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("Coverage JSON missing 'totals'.")
    covered_lines = int(totals.get("covered_lines", 0))
    num_statements = int(totals.get("num_statements", 0))
    return CoverageSummary(covered_lines=covered_lines, num_statements=num_statements)


def _extract_package_summary(payload: dict[str, object], package_name: str) -> CoverageSummary:
    """Aggregate coverage totals for one package prefix.

    Args:
        payload: Parsed coverage JSON payload.
        package_name: Package directory name under ``src/design_research_agents``.

    Returns:
        Package coverage summary.

    Raises:
        ValueError: If file-level coverage payload is malformed.
    """
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("Coverage JSON missing 'files'.")

    covered_lines = 0
    num_statements = 0
    prefix = f"src/design_research_agents/{package_name}/"
    for file_path, file_payload in files.items():
        if not isinstance(file_path, str) or not file_path.startswith(prefix):
            continue
        if not isinstance(file_payload, dict):
            continue
        summary = file_payload.get("summary")
        if not isinstance(summary, dict):
            continue
        covered_lines += int(summary.get("covered_lines", 0))
        num_statements += int(summary.get("num_statements", 0))

    return CoverageSummary(covered_lines=covered_lines, num_statements=num_statements)


def main() -> int:
    """Run coverage-threshold checks and return process status code.

    Returns:
        ``0`` when all thresholds pass, otherwise ``1``.

    Raises:
        FileNotFoundError: If the coverage report path does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-json",
        default="artifacts/coverage/coverage.json",
        help="Path to pytest-cov JSON report.",
    )
    args = parser.parse_args()

    coverage_path = Path(args.coverage_json)
    if not coverage_path.exists():
        raise FileNotFoundError(f"Coverage file not found: {coverage_path}")

    payload = _read_coverage(coverage_path)
    failures: list[str] = []

    global_summary = _extract_global_summary(payload)
    global_percent = global_summary.percent
    print(f"Global line coverage: {global_percent:.2f}% (threshold {GLOBAL_THRESHOLD:.2f}%)")
    if global_percent < GLOBAL_THRESHOLD:
        failures.append(f"global coverage {global_percent:.2f}% is below {GLOBAL_THRESHOLD:.2f}%")

    for package_name, threshold in PACKAGE_THRESHOLDS.items():
        summary = _extract_package_summary(payload, package_name)
        package_percent = summary.percent
        print(f"{package_name} line coverage: {package_percent:.2f}% (threshold {threshold:.2f}%)")
        if summary.num_statements == 0:
            failures.append(f"{package_name} coverage has no measured statements")
            continue
        if package_percent < threshold:
            failures.append(
                f"{package_name} coverage {package_percent:.2f}% is below {threshold:.2f}%"
            )

    if not failures:
        print("Coverage thresholds passed.")
        return 0

    print("Coverage threshold failures:")
    for failure in failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
