"""# Model Selection / Remote.

## Introduction
Remote model selection has the same routing tradeoffs as local selection but adds external service
variability; FrugalGPT and RouteLLM motivate policy-driven routing, and Toward Engineering AGI motivates
engineering-task-aware evaluation of those routes. This example implements remote selection with
deterministic logging.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``ModelSelector.select(...)`` with a fixed ``request_id``.
3. Evaluate model constraints and policy, then expose selector metadata in the traced payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["ModelSelector.select(...)"]
    C --> D["policy and constraints resolve one model-selection outcome"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "catalog_signature": "440e215f0fee",
     "example": "model_selection/remote.py",
     "model_id": "gpt-4o-mini",
     "policy_id": "default",
     "provider": "openai",
     "rationale": "priority=speed; selection_reason=high_load_remote; ram_budget_gb=10.0; max_latency_ms=800; gpu_p...
     "safety_constraints": {
       "max_cost_usd": null,
       "max_latency_ms": 800
     },
     "trace": {
       "request_id": "example-model-selection-remote-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162207Z_example-model-selection-remote-design-001.jsonl"
     }
   }


## References
- `FrugalGPT <https://arxiv.org/abs/2305.05176>`_
- `RouteLLM <https://arxiv.org/abs/2406.18665>`_
- `Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs <https://arxiv.org/abs/2509.16204>`_
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from design_research_agents import ModelSelector, Tracer


def _select_remote() -> dict[str, object]:
    selector = ModelSelector()
    decision = selector.select(
        task="Handle a fast design triage chat during incident response.",
        priority="speed",
        max_latency_ms=800,
        hardware_profile={
            "total_ram_gb": 16.0,
            "available_ram_gb": 12.0,
            "cpu_count": 4,
            "load_average": (6.0, 5.5, 5.0),
            "gpu_present": False,
            "gpu_vram_gb": None,
            "gpu_name": None,
            "platform_name": "example",
        },
        output="decision",
    )
    return asdict(decision)


def main() -> None:
    """Run traced remote-favoring model selection and print decision."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-model-selection-remote-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesModelSelectionRemote",
        request_id=request_id,
        input_payload={"scenario": "remote-selection"},
        function=_select_remote,
    )
    assert isinstance(payload, dict)
    payload["example"] = "model_selection/remote.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
