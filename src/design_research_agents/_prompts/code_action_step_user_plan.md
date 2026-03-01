Write Python code that solves the current step with zero or more `call_tool` calls.
Rules:
- Use only call_tool(tool_name: str, tool_input: dict).
- Use only allowed tools listed below.
- Do not write import statements.
- Do not call tools directly by name; always go through call_tool.
- If the answer depends on counting, calculation, lookup, or inspection, call a tool instead of guessing.
- Use `final_answer({...})` only when the overall task is complete.
- `final_answer({...})` is the only way to end the outer multi-step run.
- If one tool call gives enough information, you may call the tool and then `final_answer({...})` in the same step.
- You may call tools first and then call `final_answer({...})` in the same step.
- If the task is not complete yet, assign the step result to `final_output` as a dict.
- If you assign `final_output`, do not treat that as task completion.
- If you assign `final_output`, derive it from variables or tool results, not placeholder strings.
- If your last tool call returns the step result as a dict, you may set `final_output` to it.
- Return code only. No markdown. No prose.
- Example reminder (do not copy literally):
  final_output = {"result": ...}

$tools_block

User request:
$user_prompt
