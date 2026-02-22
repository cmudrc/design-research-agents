"""Run traced local-first model selection for a design summarization task.

Expected observations:
- decision indicates local-capable candidate under strict cost constraints.
- output contains provider/model rationale fields.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from design_research_agents import ModelSelector, Tracer


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
    tracer = Tracer(
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
