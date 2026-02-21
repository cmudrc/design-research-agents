"""Script tool header metadata.

@tool_name: rubric_score
@description: Score text against a simple design rubric.
@inputs:
  text: str
  max_score: int = 10
@outputs:
  stdout_json: true
@capabilities:
  filesystem_read: false
  filesystem_write: true
  network: false
  commands: []
@timeout_s: 20
@platform: [darwin, linux, windows]
@version: 1.1.0
@examples:
  - python rubric_score.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = datetime.UTC if hasattr(datetime, "UTC") else timezone(timedelta(0))


def _sanitize(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe or "example_script"


def _write_trace(*, request_id: str, payload: dict[str, object]) -> str:
    trace_dir = Path("artifacts/examples/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trace_path = trace_dir / f"run_{timestamp}_{_sanitize(request_id)}.jsonl"
    event = {
        "event_type": "ScriptToolCompleted",
        "run_id": request_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "attributes": payload,
    }
    trace_path.write_text(json.dumps(event, ensure_ascii=True) + "\n", encoding="utf-8")
    return str(trace_path)


def main() -> int:
    """Run rubric scoring over stdin JSON input and emit a tool envelope."""
    raw_input_text = sys.stdin.read()
    input_payload = json.loads(raw_input_text) if raw_input_text.strip() else {}
    rubric_text = str(input_payload.get("text", ""))
    max_score = int(input_payload.get("max_score", 10))

    score = min(max_score, max(0, len(rubric_text.split()) // 5))
    artifact_dir = Path("artifacts") / "rubric_score"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "rubric_score_report.json"
    report_payload = {
        "score": score,
        "max_score": max_score,
        "word_count": len(rubric_text.split()),
    }
    artifact_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    request_id = str(input_payload.get("request_id", "example-script-rubric-score-001"))
    trace_path = _write_trace(request_id=request_id, payload=report_payload)

    envelope = {
        "ok": True,
        "result": {
            "score": score,
            "max_score": max_score,
            "cwd": os.getcwd(),
            "trace_path": trace_path,
        },
        "artifacts": [
            {
                "path": str(artifact_path),
                "mime": "application/json",
            },
            {
                "path": trace_path,
                "mime": "application/x-ndjson",
            },
        ],
        "warnings": [],
        "error": None,
    }
    print(json.dumps(envelope, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
