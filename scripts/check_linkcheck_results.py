"""Classify Sphinx linkcheck results for deterministic CI behavior."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

NON_FAILURE_STATUSES = frozenset(
    {
        "ignored",
        "rate-limited",
        "redirected",
        "unchecked",
        "working",
    }
)
TRANSIENT_STATUSES = frozenset({"timeout"})
HARD_FAILURE_STATUSES = frozenset({"broken", "unknown"})
KNOWN_STATUSES = NON_FAILURE_STATUSES | TRANSIENT_STATUSES | HARD_FAILURE_STATUSES


@dataclass(slots=True, frozen=True)
class LinkcheckResult:
    """One normalized record from Sphinx's JSON-lines output."""

    filename: str
    lineno: int
    status: str
    uri: str
    info: str


def _load_results(path: Path) -> list[LinkcheckResult]:
    """Load and validate Sphinx linkcheck JSON-lines output.

    Args:
        path: Path to Sphinx's ``output.json`` file.

    Returns:
        Parsed linkcheck records.

    Raises:
        FileNotFoundError: If Sphinx did not create the results file.
        ValueError: If a result line is malformed or has an unknown status.
    """
    results: list[LinkcheckResult] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")

            status = payload.get("status")
            uri = payload.get("uri")
            if not isinstance(status, str) or status not in KNOWN_STATUSES:
                raise ValueError(f"Line {line_number} has unknown status {status!r}.")
            if not isinstance(uri, str) or not uri:
                raise ValueError(f"Line {line_number} has no valid URI.")

            filename = payload.get("filename", "<unknown>")
            lineno = payload.get("lineno", 0)
            info = payload.get("info", "")
            results.append(
                LinkcheckResult(
                    filename=filename if isinstance(filename, str) else "<unknown>",
                    lineno=lineno if isinstance(lineno, int) else 0,
                    status=status,
                    uri=uri,
                    info=info if isinstance(info, str) else "",
                )
            )
    return results


def _print_results(heading: str, results: list[LinkcheckResult]) -> None:
    """Print a concise group of actionable linkcheck results."""
    print(heading)
    for result in results:
        location = result.filename
        if result.lineno:
            location = f"{location}:{result.lineno}"
        detail = f" ({result.info})" if result.info else ""
        print(f"- {location}: {result.uri}{detail}")


def _classify_results(results: list[LinkcheckResult], sphinx_exit_code: int) -> int:
    """Return a stable exit code from parsed results and Sphinx's exit code."""
    hard_failures = [result for result in results if result.status in HARD_FAILURE_STATUSES]
    if hard_failures:
        _print_results("Confirmed broken links:", hard_failures)
        return 1

    transient_failures = [result for result in results if result.status in TRANSIENT_STATUSES]
    if sphinx_exit_code == 0:
        print("External linkcheck passed.")
        return 0

    if sphinx_exit_code == 1 and transient_failures:
        _print_results("Transient link timeouts (non-blocking):", transient_failures)
        print("No confirmed broken links were found.")
        return 0

    print(f"Sphinx linkcheck failed without a classifiable transient timeout (exit code {sphinx_exit_code}).")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Classify one Sphinx linkcheck run and return a process status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("docs/_build/linkcheck/output.json"),
        help="Path to Sphinx linkcheck output.json.",
    )
    parser.add_argument(
        "--sphinx-exit-code",
        type=int,
        required=True,
        help="Exit code returned by the Sphinx linkcheck builder.",
    )
    args = parser.parse_args(argv)

    try:
        results = _load_results(args.results)
    except (OSError, ValueError) as error:
        print(f"Unable to classify Sphinx linkcheck results: {error}", file=sys.stderr)
        return 1
    return _classify_results(results, args.sphinx_exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
