## Workflow Runtime Examples

These entrypoints exercise orchestration-level flows, including reusable
workflow patterns in `src/design_research_agents/workflow/implementations/`.
Most pattern classes follow agent-like reuse semantics:
initialize once, then call `.run(...)` repeatedly with per-run input.

## What Each Example Demonstrates

- `workflow_runtime.py`
  - Direct `WorkflowRuntime` orchestration with composed steps.
- `workflow_runtime_loop_step.py`
  - Direct `WorkflowRuntime` orchestration with a composable top-level `LoopStep`.
- `plan_execute.py`
  - Planner + executor pattern using runtime tools.
- `propose_critic.py`
  - Propose/critique revision loop.
- `agent_routing.py`
  - Intent/agent routing with delegate execution.
- `debate_pattern.py`
  - Structured affirmative/negative debate with judged synthesis.
- `workflow_schema_mode.py`
  - User-defined `Workflow` with `input_mode="schema"` for structured input payloads.
- `workflow_prompt_mode.py`
  - User-defined `Workflow` with `input_mode="prompt"` for string prompt payloads.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/workflow/workflow_runtime.py
PYTHONPATH=src python3 examples/workflow/workflow_runtime_loop_step.py
PYTHONPATH=src python3 examples/workflow/plan_execute.py
PYTHONPATH=src python3 examples/workflow/propose_critic.py
PYTHONPATH=src python3 examples/workflow/agent_routing.py
PYTHONPATH=src python3 examples/workflow/debate_pattern.py
PYTHONPATH=src python3 examples/workflow/workflow_schema_mode.py
PYTHONPATH=src python3 examples/workflow/workflow_prompt_mode.py
```

## Implementation Mapping

- `workflow_runtime.py` (`WorkflowRuntime`) -> `examples/workflow/workflow_runtime.py`
- `workflow_runtime_loop_step.py` (`LoopStep`) -> `examples/workflow/workflow_runtime_loop_step.py`
- `plan_execute.py` (`PlannerExecutorPattern`) -> `examples/workflow/plan_execute.py`
- `propose_critic.py` (`ReflexionPattern`) -> `examples/workflow/propose_critic.py`
- `agent_routing.py` (`RouterPattern`) -> `examples/workflow/agent_routing.py`
- `debate_pattern.py` (`DebatePattern`) -> `examples/workflow/debate_pattern.py`
- `workflow_schema_mode.py` (`Workflow` in `schema` mode) -> `examples/workflow/workflow_schema_mode.py`
- `workflow_prompt_mode.py` (`Workflow` in `prompt` mode) -> `examples/workflow/workflow_prompt_mode.py`

## Expected Outputs

- Scripts print structured workflow or runtime result payloads.
- Some flows generate artifacts under `artifacts/examples`.

## Troubleshooting

- Missing local backend dependencies:
  - Install local extras: `pip install -e '.[local]'`.
- LLM startup timeouts:
  - Increase local backend startup timeout in your LLM config if needed.
