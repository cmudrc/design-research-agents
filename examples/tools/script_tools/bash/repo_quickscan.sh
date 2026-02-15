#!/usr/bin/env bash
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
# @version: 1.0.0
# @examples:
#   - bash repo_quickscan.sh

set -euo pipefail

artifact_dir="artifacts/repo_quickscan"
mkdir -p "$artifact_dir"
report_path="$artifact_dir/report.txt"

ls -la > "$report_path"
line_count=$(wc -l < "$report_path" | tr -d ' ')

printf '{"ok":true,"result":{"line_count":%s},"artifacts":[{"path":"%s","mime":"text/plain"}],"warnings":[],"error":null}\n' "$line_count" "$report_path"
