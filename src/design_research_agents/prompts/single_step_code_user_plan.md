Write Python code that solves the request with one or more `call_tool` calls.
Rules:
- Use only call_tool(tool_name: str, tool_input: dict).
- Use only allowed tools listed below.
- Do not write import statements.
- Do not call tools directly by name; always go through call_tool.
- Assign the final result to `final_output` as a dict.
- Always end with a line that assigns `final_output`.
- If your last tool call returns the answer as a dict, set `final_output` to it.
- Return code only. No markdown. No prose.
- Required shape reminder (example only, do not copy literally):
  final_output = {"result": ...}

$tools_block

User request:
$user_prompt
