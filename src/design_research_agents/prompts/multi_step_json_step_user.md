Task: $task_prompt
Current step: $step_number
Follow a ReAct-style step:
1) Thought: infer the next best action from memory.
2) Action: choose exactly one tool and JSON input.
3) Observation: use the tool result to produce this step's output.
Return only one JSON object with this shape:
{"tool_name":"<name>","tool_input":{...},"reason":"short rationale"}
Do not include markdown or extra text.
Memory tail: $memory_tail
Retrieved context: $retrieved_context
