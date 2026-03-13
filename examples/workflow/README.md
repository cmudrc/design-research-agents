## Workflow Primitive Examples

These entrypoints focus on direct `Workflow` usage and step-level primitives.

## Scripts

- `workflow_runtime.py`
- `workflow_runtime_loop_step.py`
- `workflow_prompt_mode.py`
- `workflow_schema_mode.py`
- `workflow_model_step_design_tradeoff.py`
- `workflow_delegate_and_memory_steps.py`
- `workflow_diagram_generation.py`

## Quick Start

```bash
PYTHONPATH=src python3 examples/workflow/workflow_runtime.py
PYTHONPATH=src python3 examples/workflow/workflow_runtime_loop_step.py
PYTHONPATH=src python3 examples/workflow/workflow_delegate_and_memory_steps.py
PYTHONPATH=src python3 examples/workflow/workflow_diagram_generation.py
```

## Expected Outputs

- JSON summaries with execution order and final output details.
- Mermaid diagram text written to `artifacts/examples/workflow_diagram.mmd`.
- Trace metadata in each script payload.
