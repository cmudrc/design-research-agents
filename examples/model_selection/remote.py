"""Runnable example showing remote model selection under heavy load.

The script uses a fixed hardware profile with high load to prefer remote
selection and prints the decision.
"""

import design_research_agents as dra


def main() -> None:
    """Select a remote model when local hardware is overloaded."""
    policy = dra.models.ModelSelectionPolicy()
    intent = dra.models.ModelSelectionIntent(
        task="Handle a fast-paced live chat session.",
        priority="speed",
    )
    constraints = dra.models.ModelSelectionConstraints(max_latency_ms=800)
    hardware_profile = dra.models.HardwareProfile(
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        cpu_count=4,
        load_average=(6.0, 5.5, 5.0),
        gpu_present=False,
        gpu_vram_gb=None,
        gpu_name=None,
        platform_name="example",
    )
    decision = policy.select_model(
        intent=intent,
        constraints=constraints,
        hardware_profile=hardware_profile,
    )
    print(decision)


if __name__ == "__main__":
    main()
