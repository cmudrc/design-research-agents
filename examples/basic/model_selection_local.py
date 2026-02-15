"""Runnable example showing cost-constrained local model selection.

The script uses a fixed hardware profile and a strict cost cap to keep
selection on a local model, then prints the decision.
"""

import design_research_agents


def main() -> None:
    """Select a local model under a tight cost cap and print the decision."""
    policy = design_research_agents.ModelSelectionPolicy()
    intent = design_research_agents.ModelSelectionIntent(
        task="Summarize a research memo for stakeholders.",
        priority="quality",
    )
    # Cost cap below the remote floor keeps selection local.
    constraints = design_research_agents.ModelSelectionConstraints(max_cost_usd=0.01)
    hardware_profile = design_research_agents.HardwareProfile(
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        cpu_count=8,
        load_average=(0.2, 0.1, 0.1),
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
