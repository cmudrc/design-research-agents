Task: $task_prompt
Current step: $step_number
Follow a ReAct-style tool-routing step:
1) Think from memory.
2) Either call one route/tool, or STOP with final output.

Return JSON only with one of these shapes:
{"action":"TOOL_CALL","tool_names":["<name>","..."],"tool_input":{...},"reason":"short rationale"}
{"action":"STOP","final_output":{...},"reason":"short rationale"}

Memory tail: $memory_tail
