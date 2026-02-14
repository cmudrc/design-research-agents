"""OpenAI backend kept optional via lazy import."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OpenAIBackend:
    """Thin wrapper around the OpenAI Responses API.

    Attributes:
        model: Model name sent to the OpenAI Responses API.
        api_key_env: Environment variable name containing the API key.
        api_key: Explicit API key value. When set, it takes precedence over
            ``api_key_env``.
        base_url: Optional OpenAI-compatible server URL.
        require_api_key: Whether missing API keys should raise an error.
    """

    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    require_api_key: bool = True

    def _resolve_api_key(self) -> str:
        """Resolve the API key for this backend instance.

        Returns:
            API key string used for the OpenAI client.

        Raises:
            RuntimeError: If an API key is required but not available.
        """
        if self.api_key:
            # Explicit constructor argument takes precedence over environment.
            return self.api_key

        env_value = os.getenv(self.api_key_env)
        if env_value:
            return env_value

        if self.require_api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Use backend='echo-test' or export an API key."
            )
        # Some OpenAI-compatible local servers ignore auth and accept any token.
        return "not-needed"

    def complete(self, prompt: str) -> str:
        """Generate text using the OpenAI Responses API.

        Args:
            prompt: Prompt text sent to the model.

        Returns:
            The generated text response from OpenAI.

        Raises:
            RuntimeError: If the API key is missing, the dependency is not
                installed, or the API response is empty.
        """
        api_key = self._resolve_api_key()
        try:
            # Lazy import keeps local-only workflows lightweight.
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for backend='openai'. "
                "Install with: pip install -e ."
            ) from exc

        # Keep client options centralized so wrappers can override consistently.
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            # Enables compatibility with local OpenAI-style servers.
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        return self._complete_with_compat(client=client, prompt=prompt)

    def _complete_with_compat(self, *, client: Any, prompt: str) -> str:
        """Call OpenAI-compatible APIs with a local-server compatibility fallback.

        Args:
            client: Initialized OpenAI SDK client.
            prompt: Prompt text to generate from.

        Returns:
            Generated text.

        Raises:
            RuntimeError: If all supported request paths fail or return empty text.
        """
        try:
            # Use the Responses API as the canonical text generation path.
            response = client.responses.create(model=self.model, input=prompt)
            # `output_text` is the high-level convenience accessor in OpenAI SDK.
            output_text = str(getattr(response, "output_text", "")).strip()
            if not output_text:
                raise RuntimeError("Received an empty response from OpenAI.")
            return output_text
        except Exception as exc:
            if not self._should_use_chat_completions_fallback(exc):
                raise
            return self._complete_via_chat_completions(client=client, prompt=prompt)

    def _should_use_chat_completions_fallback(self, exc: Exception) -> bool:
        """Return whether to retry with chat completions for compatibility.

        Args:
            exc: Exception raised by the Responses API call.

        Returns:
            ``True`` when the backend should retry with chat completions.
        """
        # Only OpenAI-compatible base URLs are expected to miss `/v1/responses`.
        if not self.base_url:
            return False
        if getattr(exc, "status_code", None) != 404:
            return False
        return self._looks_like_missing_responses_endpoint(exc)

    def _looks_like_missing_responses_endpoint(self, exc: Exception) -> bool:
        """Return whether a 404 appears tied to the `/responses` endpoint.

        Local OpenAI-compatible servers can return 404 when they do not expose
        ``/v1/responses``. We only retry with chat completions in that case.
        """
        # Prefer structured request metadata when available.
        request = getattr(exc, "request", None)
        request_url = getattr(request, "url", None)
        if _url_contains_responses_path(request_url):
            return True

        response = getattr(exc, "response", None)
        response_url = getattr(response, "url", None)
        if _url_contains_responses_path(response_url):
            return True

        # Fallback to message heuristics when structured fields are absent.
        message = str(exc).lower()
        if "/responses" in message:
            return True

        # Some compatibility shims raise bare 404s with no detail text.
        return message == ""

    def _complete_via_chat_completions(self, *, client: Any, prompt: str) -> str:
        """Generate text via chat completions for OpenAI-compatible servers.

        Args:
            client: Initialized OpenAI SDK client.
            prompt: Prompt text to send as a user message.

        Returns:
            Generated text extracted from the first choice.

        Raises:
            RuntimeError: If the chat completions response has no usable text.
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("Received an empty chat completions response from OpenAI.")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        output_text = self._extract_text_content(content)
        if not output_text:
            raise RuntimeError("Received an empty response from OpenAI chat completions.")
        return output_text

    def _extract_text_content(self, content: Any) -> str:
        """Extract plain text from chat completions content payloads.

        Args:
            content: Message content from chat completions.

        Returns:
            Extracted text, or an empty string when no text is found.
        """
        if isinstance(content, str):
            return content.strip()

        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
                continue
            text_attr = getattr(part, "text", None)
            if isinstance(text_attr, str):
                text_parts.append(text_attr)

        return "\n".join(piece.strip() for piece in text_parts if piece.strip())


def complete(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    api_key: str | None = None,
    base_url: str | None = None,
    require_api_key: bool = True,
) -> str:
    """Generate text with OpenAI.

    Args:
        prompt: Prompt text sent to the OpenAI backend.
        model: Model name sent to the API.
        api_key_env: Environment variable used to resolve API keys.
        api_key: Explicit API key value, if not using an environment variable.
        base_url: Optional OpenAI-compatible API base URL.
        require_api_key: Whether missing API keys should raise an error.

    Returns:
        Generated response text from :class:`OpenAIBackend`.

    Raises:
        RuntimeError: If backend requirements are not met or generation fails.
    """
    backend = OpenAIBackend(
        model=model,
        api_key_env=api_key_env,
        api_key=api_key,
        base_url=base_url,
        require_api_key=require_api_key,
    )
    return backend.complete(prompt)


def _url_contains_responses_path(raw_url: object) -> bool:
    """Return whether a URL-like object points at the Responses endpoint."""
    if raw_url is None:
        return False
    normalized = str(raw_url).strip().lower()
    if not normalized:
        return False
    return "/responses" in normalized
