"""Materialize one built-in knowledge profile into local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from design_research_agents.memory import (
    SQLiteMemoryStore,
    load_builtin_knowledge_profile,
    seed_builtin_knowledge_profile,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Built-in knowledge profile name.")
    parser.add_argument(
        "--namespace",
        default="default",
        help="Namespace used when seeding stores. Defaults to 'default'.",
    )
    parser.add_argument(
        "--sqlite-db",
        type=Path,
        default=None,
        help="Optional SQLite database path to seed with the profile's memory records.",
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        default=None,
        help="Optional JSON path for the profile's graph nodes and edges.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON path for a summary manifest of the materialized profile.",
    )
    return parser


def main() -> None:
    """Build one knowledge profile manifest and optional SQLite seed."""
    args = build_parser().parse_args()
    profile = load_builtin_knowledge_profile(args.profile)

    summary = {
        "profile": profile.to_dict(),
        "namespace": args.namespace.strip() or "default",
        "memory_records_written": 0,
    }

    if args.sqlite_db is not None:
        args.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
        with SQLiteMemoryStore(db_path=args.sqlite_db) as store:
            seed_result = seed_builtin_knowledge_profile(
                profile.name,
                memory_store=store,
                namespace=args.namespace,
            )
        summary["memory_records_written"] = seed_result.memory_records_written
        summary["sqlite_db"] = str(args.sqlite_db)

    if args.graph_json is not None:
        args.graph_json.parent.mkdir(parents=True, exist_ok=True)
        args.graph_json.write_text(
            json.dumps(
                {
                    "profile": profile.name,
                    "namespace": summary["namespace"],
                    "nodes": [node.to_dict() for node in profile.graph_nodes],
                    "edges": [edge.to_dict() for edge in profile.graph_edges],
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summary["graph_json"] = str(args.graph_json)

    rendered_summary = json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(rendered_summary, encoding="utf-8")
    else:
        print(rendered_summary, end="")


if __name__ == "__main__":
    main()
