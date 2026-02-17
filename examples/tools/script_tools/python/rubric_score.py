"""Script tool header metadata.

@tool_name: rubric_score
@description: Score text against a simple rubric.
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
@version: 1.0.0
@examples:
  - python rubric_score.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    """Run rubric scoring over stdin JSON input and emit a tool envelope.

    Returns:
        The resulting value.
    """
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
    artifact_path.write_text(
        json.dumps(report_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    envelope = {
        "ok": True,
        "result": {
            "score": score,
            "max_score": max_score,
            "cwd": os.getcwd(),
        },
        "artifacts": [
            {
                "path": str(artifact_path),
                "mime": "application/json",
            }
        ],
        "warnings": [],
        "error": None,
    }
    print(json.dumps(envelope, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
