"""Local fallback backend with an OpenAI-like surface."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalOpenAIBackend:
    """Deterministic backend used for local development and CI.

    Attributes:
        model: Label included in the generated response prefix.
    """

    model: str = "local-echo"

    def complete(self, prompt: str) -> str:
        """Generate a deterministic local response.

        Args:
            prompt: Prompt text to transform into a local response.

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
    """Generate text using the local backend.

    Args:
        prompt: Prompt text for the local backend.

    Returns:
        Generated response from :class:`LocalOpenAIBackend`.
    """
    return LocalOpenAIBackend().complete(prompt)
