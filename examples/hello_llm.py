"""Minimal runnable example that calls ``complete`` directly."""

from __future__ import annotations

from design_research_agents import complete


def main() -> None:
    """Call ``complete`` and print the response."""
    # Keep example minimal: one prompt, one backend, one print.
    prompt = "Say hello from design-research-agents."
    response = complete(prompt, backend="local")
    print(response)


if __name__ == "__main__":
    main()
