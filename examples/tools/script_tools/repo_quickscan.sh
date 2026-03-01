#!/usr/bin/env bash
# # Script Tools / Repo Quickscan.
#
# ## Introduction
# This script focuses on the practical so-what for tool integration: a deterministic, inspectable JSON
# contract for repository quickscan operations.
#
# ## Technical Implementation
# 1. Read JSON input from ``stdin`` and execute the quickscan workflow.
# 2. Write report + trace artifacts under ``artifacts/``.
# 3. Emit one JSON envelope on ``stdout`` with ``ok/result/artifacts/warnings/error``.
#
# ```mermaid
# flowchart LR
# A["JSON input payload"] --> B["repo_quickscan.sh"]
# B --> C["report.txt + trace artifact"]
# C --> D["JSON envelope on stdout"]
# ```
#
# ## Expected Results
# - Reads one JSON object from ``stdin``.
# - Prints one JSON envelope containing ``ok/result/artifacts``.
# - Writes artifacts under ``artifacts/repo_quickscan`` and ``artifacts/examples/traces``.

set -euo pipefail

artifact_dir="artifacts/repo_quickscan"
mkdir -p "$artifact_dir"
report_path="$artifact_dir/report.txt"

ls -la > "$report_path"
line_count=$(wc -l < "$report_path" | tr -d ' ')

trace_dir="artifacts/examples/traces"
mkdir -p "$trace_dir"
request_id="example-script-repo-quickscan-001"
timestamp=$(date -u +"%Y%m%dT%H%M%SZ")
safe_request_id=$(printf '%s' "$request_id" | tr -c 'A-Za-z0-9_-' '_')
trace_path="$trace_dir/run_${timestamp}_${safe_request_id}.jsonl"
printf '{"event_type":"ScriptToolCompleted","run_id":"%s","attributes":{"line_count":%s}}\n' "$request_id" "$line_count" > "$trace_path"

printf '{"ok":true,"result":{"line_count":%s,"trace_path":"%s"},"artifacts":[{"path":"%s","mime":"text/plain"},{"path":"%s","mime":"application/x-ndjson"}],"warnings":[],"error":null}\n' "$line_count" "$trace_path" "$report_path" "$trace_path"
