Task: $task_prompt
Current step: $step_number
Choose one internal action:
1) CONTINUE
2) STOP

Return JSON only with one of these shapes:
{"decision":"CONTINUE","content":"refined partial answer or next subgoal","reason":"short rationale"}
{"decision":"STOP","content":"final answer summary","final_output":"final answer text","reason":"short rationale"}

Memory tail: $memory_tail
