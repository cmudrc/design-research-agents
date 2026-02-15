"""Discovery for manifest-less lazy tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .parser import LazyHeaderError, LazyToolHeader, parse_lazy_tool_header


@dataclass(slots=True, frozen=True)
class LazyDiscoveryDiagnostic:
    """Diagnostic emitted when one lazy-tool script fails header parsing."""

    path: str
    error: str


@dataclass(slots=True, frozen=True)
class LazyToolDefinition:
    """Resolved lazy-tool definition including source path and parsed header."""

    path: str
    header: LazyToolHeader


def discover_lazy_tools(
    search_paths: tuple[str, ...],
) -> tuple[list[LazyToolDefinition], list[LazyDiscoveryDiagnostic]]:
    """Discover lazy tools from configured search paths."""
    found_tools: list[LazyToolDefinition] = []
    diagnostics: list[LazyDiscoveryDiagnostic] = []

    for raw_path in search_paths:
        base = Path(raw_path).expanduser()
        if not base.exists():
            continue
        candidates: list[Path] = []
        if base.is_file():
            candidates.append(base)
        else:
            candidates.extend(base.rglob("*.py"))
            candidates.extend(base.rglob("*.sh"))

        for candidate in sorted(candidates):
            if candidate.suffix not in {".py", ".sh"}:
                continue
            try:
                header = parse_lazy_tool_header(candidate)
            except LazyHeaderError as exc:
                diagnostics.append(LazyDiscoveryDiagnostic(path=str(candidate), error=str(exc)))
                continue
            found_tools.append(LazyToolDefinition(path=str(candidate.resolve()), header=header))

    return found_tools, diagnostics


__all__ = ["LazyDiscoveryDiagnostic", "LazyToolDefinition", "discover_lazy_tools"]
