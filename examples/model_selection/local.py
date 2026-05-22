"""# Model Selection / Local.

## Introduction
FrugalGPT and RouteLLM both frame model selection as a policy problem balancing capability, latency, and
cost, while HELM stresses evaluation rigor across model choices. This example demonstrates local model
selection policy execution with observable scoring outputs and traces.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build a default ``ModelFlightRegistry`` and flatten it into the ``ModelCatalog`` passed to ``ModelSelector``.
3. Execute ``ModelSelector.select(...)`` with a fixed ``request_id``.
4. Evaluate model constraints and policy, then expose selector metadata in the traced payload.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

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
     "catalog_signature": "4dbd48aeadb6",
     "example": "model_selection/local.py",
     "model_id": "llama-3.1-8b-instruct-gguf-q4_k_m",
     "policy_id": "default",
     "provider": "llama_cpp",
     "rationale": "priority=quality; selection_reason=local_fit; model_size_b=8.0; ram_budget_gb=10.0; max_cost_us...
     "safety_constraints": {
       "max_cost_usd": 0.01,
       "max_latency_ms": null
     },
     "trace": {
       "request_id": "example-model-selection-local-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162207Z_example-model-selection-local-design-001.jsonl"
     }
   }


## References
- `FrugalGPT <https://arxiv.org/abs/2305.05176>`_
- `RouteLLM <https://arxiv.org/abs/2406.18665>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import design_research_agents as drag


def _default_model_catalog() -> drag.ModelCatalog:
    flight_registry = drag.ModelFlightRegistry.default()
    flights: tuple[drag.ModelFlight, ...] = tuple(flight_registry.flights)
    return drag.ModelCatalog.from_flights(flights)


def _select_local() -> dict[str, object]:
    selector = drag.ModelSelector(catalog=_default_model_catalog())
    decision = selector.select(
        task="Summarize engineering design review findings for stakeholders.",
        priority="quality",
        max_cost_usd=0.01,
        hardware_profile={
            "total_ram_gb": 16.0,
            "available_ram_gb": 12.0,
            "cpu_count": 8,
            "load_average": (0.2, 0.1, 0.1),
            "gpu_present": False,
            "gpu_vram_gb": None,
            "gpu_name": None,
            "platform_name": "example",
        },
        output="decision",
    )
    return asdict(decision)


def main() -> None:
    """Run traced local-first model selection and print decision."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-model-selection-local-design-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesModelSelectionLocal",
        request_id=request_id,
        input_payload={"scenario": "local-selection"},
        function=_select_local,
    )
    assert isinstance(payload, dict)
    payload["example"] = "model_selection/local.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
