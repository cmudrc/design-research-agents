Decide whether the agent should continue to another action step.
Return JSON only: {"continue": <bool>, "thought": "..."}.
Decision policy:
- If no observation exists yet, set continue=true.
- If latest observation failed, set continue=false.
- If task is fully satisfied from memory set continue=false; otherwise continue=true.
- `thought` should capture the short ReAct rationale for this decision.

Step number: $step_number
Task: $task_prompt
Memory tail: $memory_tail
Retrieved context: $retrieved_context
