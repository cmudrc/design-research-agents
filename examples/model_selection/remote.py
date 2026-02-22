"""Example script.

Motivation
Run traced remote-favoring model selection under heavy local load.

Diagram
```mermaid
flowchart LR
    A["Task profile"] --> B["Model selector"]
    B --> C["remote decision"]
    C --> D["Decision payload and trace"]
```

Technical Walkthrough
1. Configure the runtime surface for `model_selection` use-cases and run `remote`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/model_selection/remote.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
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
