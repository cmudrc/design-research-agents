"""Filesystem core tools with policy-guarded read/write behavior."""

from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Mapping

from design_research_agents.contracts.tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools.policy import ToolPolicy
from design_research_agents.tools.sources.inprocess_source import InProcessToolSource

from ._helpers import get_bool, get_int, get_str


def register_fs_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register policy-guarded filesystem read/write utility tools.

    Args:
        source: Parameter value.
        policy: Parameter value.
    """
    read_metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(filesystem_read=True),
        timeout_s=10,
        max_output_bytes=65_536,
        risky=False,
    )
    write_metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(filesystem_write=True),
        timeout_s=10,
        max_output_bytes=65_536,
        risky=True,
    )

    source.register_tool(
        spec=ToolSpec(
            name="fs.list_dir",
            description="List entries in a directory.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "max_entries": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=read_metadata,
        ),
        handler=lambda i, r, d: _list_dir(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="fs.read_text",
            description="Read UTF-8 text from a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=read_metadata,
        ),
        handler=lambda i, r, d: _read_text(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="fs.write_text",
            description="Write text to a file, restricted to artifacts by default.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=write_metadata,
        ),
        handler=lambda i, r, d: _write_text(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="fs.glob",
            description="Return file paths matching a glob pattern.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["path", "pattern"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=read_metadata,
        ),
        handler=lambda i, r, d: _glob(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="fs.stat",
            description="Return metadata for one path.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=read_metadata,
        ),
        handler=lambda i, r, d: _stat(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="fs.hash",
            description="Compute a file hash.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "algo": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=read_metadata,
        ),
        handler=lambda i, r, d: _hash(i, policy=policy),
    )


def _list_dir(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Run list dir.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.
    """
    path = policy.resolve_read_path(get_str(input_dict, "path", default="."))
    pattern = get_str(input_dict, "pattern", default="*")
    max_entries = get_int(input_dict, "max_entries", default=200)

    entries: list[dict[str, object]] = []
    for entry in sorted(path.iterdir(), key=lambda item: item.name)[:max_entries]:
        if not fnmatch.fnmatch(entry.name, pattern):
            continue
        stat = entry.stat()
        entries.append(
            {
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
                "size": stat.st_size,
            }
        )
    return {"path": str(path), "entries": entries, "count": len(entries)}


def _read_text(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Run read text.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.
    """
    path = policy.resolve_read_path(get_str(input_dict, "path"))
    max_bytes = get_int(input_dict, "max_bytes", default=65_536)

    raw_bytes = path.read_bytes()
    truncated = False
    if len(raw_bytes) > max_bytes:
        raw_bytes = raw_bytes[:max_bytes]
        truncated = True
    text = raw_bytes.decode("utf-8", errors="replace")

    return {
        "path": str(path),
        "text": text,
        "truncated": truncated,
        "size_bytes": len(raw_bytes),
    }


def _write_text(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Run write text.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    path = policy.resolve_write_path(get_str(input_dict, "path"))
    content = get_str(input_dict, "content")
    overwrite = get_bool(input_dict, "overwrite", default=False)

    if path.exists() and not overwrite:
        raise ValueError(f"Path already exists: {path}. Pass overwrite=true to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "bytes_written": len(content.encode("utf-8")),
    }


def _glob(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Run glob.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.
    """
    root = policy.resolve_read_path(get_str(input_dict, "path"))
    pattern = get_str(input_dict, "pattern")
    matches = sorted(str(path) for path in root.rglob(pattern))
    return {
        "root": str(root),
        "pattern": pattern,
        "matches": matches,
        "count": len(matches),
    }


def _stat(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Run stat.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.
    """
    path = policy.resolve_read_path(get_str(input_dict, "path"))
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "mode": stat.st_mode,
    }


def _hash(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Run hash.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    path = policy.resolve_read_path(get_str(input_dict, "path"))
    algo = get_str(input_dict, "algo", default="sha256").lower()
    if algo not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm '{algo}'.")
    hasher = hashlib.new(algo)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8_192)
            if not chunk:
                break
            hasher.update(chunk)
    return {
        "path": str(path),
        "algo": algo,
        "digest": hasher.hexdigest(),
    }
