"""Lightweight data utility tools."""

from __future__ import annotations

import csv
from collections.abc import Mapping

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._policy import ToolPolicy
from design_research_agents.tools._sources._inprocess_source import InProcessToolSource

from ._helpers import get_int, get_str


def register_data_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register CSV loading and summary tools with policy-guarded reads.

    Args:
        source: Input value for this parameter.
        policy: Input value for this parameter.
    """
    metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(filesystem_read=True),
        timeout_s=15,
        max_output_bytes=131_072,
        risky=False,
    )
    source.register_tool(
        spec=ToolSpec(
            name="data.load_csv",
            description="Load CSV rows from a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "nrows": {"type": "integer"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=metadata,
        ),
        handler=lambda i, r, d: _load_csv(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="data.describe",
            description="Describe basic CSV stats.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "kind": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=metadata,
        ),
        handler=lambda i, r, d: _describe(i, policy=policy),
    )


def _load_csv(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Run load csv.

    Args:
        input_dict: Input value for this parameter.
        policy: Input value for this parameter.

    Returns:
        Computed return value.
    """
    path = policy.resolve_read_path(get_str(input_dict, "path"))
    nrows = get_int(input_dict, "nrows", default=100)
    columns_raw = input_dict.get("columns")
    columns = [str(item) for item in columns_raw] if isinstance(columns_raw, list) else None

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if columns is not None:
                rows.append({key: row.get(key) for key in columns})
            else:
                rows.append(dict(row))
            if index + 1 >= nrows:
                break

    return {
        "path": str(path),
        "rows": rows,
        "count": len(rows),
    }


def _describe(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Describe tabular data using lightweight CSV summary statistics.

    Args:
        input_dict: Input value for this parameter.
        policy: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    path = policy.resolve_read_path(get_str(input_dict, "path"))
    kind = get_str(input_dict, "kind", default="csv").lower()
    if kind != "csv":
        raise ValueError("Only kind='csv' is supported in v1.")

    row_count = 0
    columns: list[str] = []
    missing_by_column: dict[str, int] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing_by_column = {column: 0 for column in columns}
        for row in reader:
            row_count += 1
            for column in columns:
                value = row.get(column)
                if value is None or value == "":
                    missing_by_column[column] += 1

    return {
        "path": str(path),
        "kind": kind,
        "rows": row_count,
        "columns": columns,
        "column_count": len(columns),
        "missing_by_column": missing_by_column,
    }
