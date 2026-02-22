#!/usr/bin/env bash
# Motivation
# Produce a quick repository inventory as a runnable shell-script tool example.
#
# Diagram
# ```mermaid
# flowchart LR
#     A["Tool input"] --> B["Shell quickscan"]
#     B --> C["Report artifact and trace"]
# ```
#
# Technical Walkthrough
# 1. Read repository directory metadata via `ls -la`.
# 2. Write a plain-text artifact and emit a deterministic trace record.
# 3. Print one JSON tool envelope to stdout.
#
# Expected Results
# - The script exits successfully and prints one JSON object.
# - The output includes `line_count` and `trace_path`.
# - Artifacts are written under `artifacts/repo_quickscan` and `artifacts/examples/traces`.
#
# Discussion
# This script is capability-first; deterministic behavior for full examples runs is handled in tests via monkeypatching, not by branching in this script.
#
# @tool_name: repo_quickscan
# @description: Produce a quick repository inventory snapshot.
# @inputs:
#   include_hidden: bool = false
# @outputs:
#   stdout_json: true
# @capabilities:
#   filesystem_read: true
#   filesystem_write: true
#   network: false
#   commands: []
# @timeout_s: 20
# @platform: [darwin, linux]
# @version: 1.1.0
# @examples:
#   - bash repo_quickscan.sh

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
