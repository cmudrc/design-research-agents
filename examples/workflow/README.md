## Workflow Examples

These entrypoints cover traced orchestration flows and reusable patterns for
engineering-design tasks.

## Coverage Highlights

- Public patterns: `ConversationPattern`, `DebatePattern`, `PlannerExecutorPattern`,
  `ReflexionPattern`, `RouterPattern`, `NetworkedPattern`, `BlackboardPattern`,
  `TreeSearchPattern`, `RagReasoningPattern`.
- Public workflow step classes demonstrated directly, including:
  `LogicStep`, `ToolStep`, `AgentStep`, `LoopStep`, `ModelStep`,
  `DelegateBatchStep`, `MemoryReadStep`, `MemoryWriteStep`.

## Scripts

- `workflow_runtime.py`
- `workflow_runtime_loop_step.py`
- `workflow_prompt_mode.py`
- `workflow_schema_mode.py`
- `workflow_model_step_design_tradeoff.py`
- `workflow_delegate_and_memory_steps.py`
- `plan_execute.py`
- `propose_critic.py`
- `agent_routing.py`
- `debate_pattern.py`
- `conversation_pattern.py`
- `networked_blackboard.py`
- `tree_search.py`
- `rag_reasoning.py`

## Quick Start

```bash
PYTHONPATH=src python3 examples/workflow/workflow_runtime.py
PYTHONPATH=src python3 examples/workflow/workflow_model_step_design_tradeoff.py
PYTHONPATH=src python3 examples/workflow/workflow_delegate_and_memory_steps.py
```

## Expected Outputs

- Each script prints run summaries with key orchestration fields.
- All scripts include trace metadata.
