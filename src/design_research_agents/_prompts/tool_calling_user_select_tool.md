Select exactly one action from the list and provide JSON arguments.
`final_answer` is always available as a built-in action.
Choose a real tool when you still need to gather information or do work.
Choose `final_answer` only when you are ready to end the task.
Return only one JSON object with this shape:
{"tool_name":"<name>","tool_input":{...},"reason":"short rationale"}
Do not include markdown or extra text.

$choices_block

User request:
$user_prompt
