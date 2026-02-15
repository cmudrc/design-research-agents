"""Runnable example showing remote model selection under heavy load.

The script uses a fixed hardware profile with high load to prefer remote
selection and prints the decision.
"""

from design_research_agents import ModelSelector


def main() -> None:
    """Select a remote model when local hardware is overloaded."""
    selector = ModelSelector()
    decision = selector.select(
        task="Handle a fast-paced live chat session.",
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
    print(decision)


if __name__ == "__main__":
    main()
