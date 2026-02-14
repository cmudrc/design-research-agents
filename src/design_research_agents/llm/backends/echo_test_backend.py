"""Deterministic echo backend used for tests and local smoke checks.

This backend intentionally avoids any network or model dependency and returns
normalized prompt echoes so higher-level behavior can be tested deterministically.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EchoTestBackend:
    """Deterministic echo backend for tests and local smoke checks.

    Attributes:
        model: Label included in the generated response prefix.
    """

    model: str = "echo-test"

    def complete(self, prompt: str) -> str:
        """Generate a deterministic echo response.

        Args:
            prompt: Prompt text to transform into an echo response.

        Returns:
            A normalized response string prefixed with the backend model name.
        """
        # Normalize whitespace so test outputs are stable across input styles.
        cleaned_prompt = " ".join(prompt.strip().split())
        if not cleaned_prompt:
            # Keep empty prompts useful for demos and smoke tests.
            cleaned_prompt = "Hello from design-research-agents."
        return f"[{self.model}] {cleaned_prompt}"


def complete(prompt: str) -> str:
    """Generate text using the echo-test backend.

    Args:
        prompt: Prompt text for the echo-test backend.

    Returns:
        Generated response from :class:`EchoTestBackend`.
    """
    return EchoTestBackend().complete(prompt)
