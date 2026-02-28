Task: $task_prompt
Current step: $step_number
Follow a ReAct-style step:
1) Thought: infer the next best action from memory.
2) Action: write Python that uses zero or more allowed tools.
3) Observation: use tool results to produce this step's output.
Return Python code only.
Use tools only via call_tool("tool_name", {...}).
Do not use import statements.
Use final_answer({...}) when you are ready to end the overall task.
Use final_output = {...} only when recording this step's observation and continuing.
You may call tools first and then call final_answer({...}) in the same step.
Memory tail: $memory_tail
Retrieved context: $retrieved_context
