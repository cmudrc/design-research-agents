Select exactly one action from the list and provide JSON arguments.
`final_answer` is always available as a built-in action.
Choose a real tool when you still need to gather information or do work.
If the answer depends on counting, calculation, lookup, or inspection, choose a real tool instead of guessing.
Choose `final_answer` only when you are ready to end the task.
Do not use `final_answer` to guess values that should come from prior tool observations.
Return only one JSON object with this shape:
{"tool_name":"<name>","tool_input":{...},"reason":"short rationale"}
Do not include markdown or extra text.

$choices_block

User request:
$user_prompt
