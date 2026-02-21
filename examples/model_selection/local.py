"""Run traced local-first model selection for a design summarization task.

Expected observations:
- decision indicates local-capable candidate under strict cost constraints.
- output contains provider/model rationale fields.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from dataclasses import asdict

from design_research_agents import ModelSelector
from design_research_agents.shared.example_support import (
    print_json,
    run_traced_callable,
    trace_info,
)


def _select_local() -> dict[str, object]:
    selector = ModelSelector()
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
    request_id = "example-model-selection-local-design-001"
    payload = run_traced_callable(
        agent_name="ExamplesModelSelectionLocal",
        request_id=request_id,
        input_payload={"scenario": "local-selection"},
        function=_select_local,
    )
    assert isinstance(payload, dict)
    payload["example"] = "model_selection/local.py"
    payload["trace"] = trace_info(request_id)
    print_json(payload)


if __name__ == "__main__":
    main()
