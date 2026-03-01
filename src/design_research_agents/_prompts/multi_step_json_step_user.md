Task: $task_prompt
Current step: $step_number
Follow a ReAct-style step:
1) Thought: infer the next best action from memory.
2) Action: choose exactly one action and JSON input.
3) Observation: use the action result to produce this step's output.
Choose a real tool if more work is needed.
If the answer depends on counting, calculation, lookup, or inspection, use a real tool instead of guessing.
Choose `final_answer` if the prior observations are enough to finish the task.
Do not use `final_answer` to guess values that should come from prior tool observations.
Return only one JSON object with this shape:
{"tool_name":"<name>","tool_input":{...},"reason":"short rationale"}
Do not include markdown or extra text.
Memory tail: $memory_tail
Retrieved context: $retrieved_context
