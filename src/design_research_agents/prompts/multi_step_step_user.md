Task: $task_prompt
Current step: $step_number
Follow a ReAct-style step:
1) Thought: infer the next best action from memory.
2) Action: write Python that uses one or more allowed tools.
3) Observation: use tool results to produce this step's output.
Return Python code only.
Use tools only via call_tool("tool_name", {...}).
Do not use import statements.
Assign final_output to a dict before finishing.
Memory tail: $memory_tail
Retrieved context: $retrieved_context
