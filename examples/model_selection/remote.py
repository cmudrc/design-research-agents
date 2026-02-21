"""Run traced remote-favoring model selection under heavy local load.

Expected observations:
- decision reflects remote-capable choice when local load is constrained.
- rationale explains speed/latency policy tradeoff.
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
    request_id = "example-model-selection-remote-design-001"
    payload = run_traced_callable(
        agent_name="ExamplesModelSelectionRemote",
        request_id=request_id,
        input_payload={"scenario": "remote-selection"},
        function=_select_remote,
    )
    assert isinstance(payload, dict)
    payload["example"] = "model_selection/remote.py"
    payload["trace"] = trace_info(request_id)
    print_json(payload)


if __name__ == "__main__":
    main()
