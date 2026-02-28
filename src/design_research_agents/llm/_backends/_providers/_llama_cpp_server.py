"""Managed backend for running a local ``llama_cpp.server`` process.

The backend handles server lifecycle and readiness polling. Completion calls
are handled by higher-level backends that target the exposed OpenAI-compatible
HTTP endpoint.
"""

from __future__ import annotations

import atexit
import importlib
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


@dataclass(slots=True, kw_only=True)
class LlamaCppServerBackend:
    """Manage a persistent llama-cpp server process.

    Attributes:
        model: ``llama_cpp.server`` ``--model`` value (path or filename/pattern).
        hf_model_repo_id: Optional Hugging Face repo id for remote model loading.
        api_model: OpenAI-compatible model alias exposed by the local server.
        host: Server bind host.
        port: Server bind port.
        startup_timeout_seconds: Maximum wait time for server startup.
        poll_interval_seconds: Delay between readiness checks.
        python_executable: Python executable used to launch the server process.
        extra_server_args: Additional CLI arguments for ``llama_cpp.server``.
    """

    model: str
    """Stored ``model`` value."""
    hf_model_repo_id: str | None = None
    """Stored ``hf_model_repo_id`` value."""
    api_model: str = "local-model"
    """Stored ``api_model`` value."""
    host: str = "127.0.0.1"
    """Stored ``host`` value."""
    port: int = 8001
    """Stored ``port`` value."""
    startup_timeout_seconds: float = 60.0
    """Stored ``startup_timeout_seconds`` value."""
    poll_interval_seconds: float = 0.25
    """Stored ``poll_interval_seconds`` value."""
    python_executable: str = sys.executable
    """Stored ``python_executable`` value."""
    extra_server_args: tuple[str, ...] = ()
    """Stored ``extra_server_args`` value."""
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    """Stored ``_process`` value."""

    def __post_init__(self) -> None:
        """Initialize delegated OpenAI-compatible caller and shutdown hook.

        Raises:
            ValueError: If ``model`` is empty.
        """
        model_value = self.model.strip()
        if not model_value:
            raise ValueError("model must not be empty.")
        self.model = model_value
        api_model_value = self.api_model.strip()
        if not api_model_value:
            raise ValueError("api_model must not be empty.")
        self.api_model = api_model_value
        if self.hf_model_repo_id is not None:
            # Empty strings are treated as "not provided".
            self.hf_model_repo_id = self.hf_model_repo_id.strip() or None

        # Best-effort cleanup for normal interpreter shutdown.
        atexit.register(self.close)

    @property
    def base_url(self) -> str:
        """Return OpenAI-compatible API base URL exposed by this server.

        This URL is used for readiness checks and by caller backends that send
        completion requests.

        Returns:
            OpenAI-compatible base URL string.
        """
        return f"http://{self.host}:{self.port}/v1"

    def _build_command(self) -> list[str]:
        """Build startup command used to launch ``llama_cpp.server``.

        Includes model aliasing and optional Hugging Face repository arguments.

        Returns:
            Command list suitable for ``subprocess.Popen``.
        """
        # Launch the packaged server module so config stays Python-environment local.
        command = [
            self.python_executable,
            "-m",
            "llama_cpp.server",
            "--model",
            self.model,
            "--model_alias",
            self.api_model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.hf_model_repo_id:
            # When set, the server resolves --model relative to this HF repo.
            command.extend(["--hf_model_repo_id", self.hf_model_repo_id])
        command.extend(self.extra_server_args)
        return command

    def is_running(self) -> bool:
        """Return whether managed server subprocess is currently alive.

        A process is considered alive when the handle exists and has not exited.

        Returns:
            ``True`` when the server process is alive, otherwise ``False``.
        """
        return self._process is not None and self._process.poll() is None

    def _ensure_server_dependency(self) -> None:
        """Validate that ``llama_cpp.server`` is available in this environment.

        Raises:
            RuntimeError: If required llama-cpp server dependencies are missing.
        """
        try:
            spec = find_spec("llama_cpp.server")
        except ModuleNotFoundError:
            spec = None

        if spec is None:
            raise RuntimeError("llama-cpp server dependency is missing. Install with: pip install -e '.[llama_cpp]'")

        if self.hf_model_repo_id and find_spec("huggingface_hub") is None:
            raise RuntimeError(
                "huggingface-hub is required when hf_model_repo_id is set. Install with: pip install -e '.[llama_cpp]'"
            )
        self._resolve_hf_model_name()

    def _resolve_hf_model_name(self) -> None:
        """Resolve best-effort HF GGUF filename when a repo id is provided.

        This keeps short model names ergonomic (for example
        ``tinyllama.Q4_K_M.gguf``) by mapping them to an actual file in the
        configured repository when there is a unique match.
        """
        if not self.hf_model_repo_id:
            return

        requested_model = self.model
        if "/" in requested_model:
            # Paths are assumed to be explicit and should not be rewritten.
            return

        try:
            huggingface_hub = importlib.import_module("huggingface_hub")
            list_repo_files = cast(
                Callable[[str], Sequence[str]],
                huggingface_hub.list_repo_files,
            )
        except (ImportError, AttributeError, TypeError):
            return

        try:
            repo_files = list_repo_files(self.hf_model_repo_id)
        except Exception:
            # Keep startup behavior deterministic even if HF lookup fails.
            return

        gguf_files = [filename for filename in repo_files if filename.lower().endswith(".gguf")]
        if not gguf_files:
            return

        if requested_model in gguf_files:
            return

        lowercase_files = {filename.lower(): filename for filename in gguf_files}
        exact_lower_match = lowercase_files.get(requested_model.lower())
        if exact_lower_match:
            self.model = exact_lower_match
            return

        # Try a direct suffix match first.
        suffix_matches = [filename for filename in gguf_files if filename.lower().endswith(requested_model.lower())]
        if len(suffix_matches) == 1:
            self.model = suffix_matches[0]
            return

        # Then match by quantization suffix like ``Q4_K_M.gguf``.
        quant_suffix_match = re.search(r"(q\d(?:_[a-z0-9]+)*\.gguf)$", requested_model.lower())
        if not quant_suffix_match:
            return

        quant_suffix = quant_suffix_match.group(1)
        quant_matches = [filename for filename in gguf_files if filename.lower().endswith(quant_suffix)]
        if len(quant_matches) == 1:
            self.model = quant_matches[0]

    def start(self) -> None:
        """Start the llama-cpp server if it is not already running.

        Raises:
            RuntimeError: If the server fails to launch or become ready.
        """
        if self.is_running():
            return

        # Fail fast with a direct dependency message before spawning a process.
        self._ensure_server_dependency()

        self._process = subprocess.Popen(
            self._build_command(),
            # Avoid flooding stdout in long runs; errors still bubble via health checks.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        """Wait until the OpenAI-compatible health endpoint responds.

        Raises:
            RuntimeError: If startup times out or the process exits early.
        """
        deadline = time.monotonic() + self.startup_timeout_seconds
        # `/models` is a lightweight OpenAI-compatible readiness probe.
        health_url = f"{self.base_url}/models"
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                raise RuntimeError("llama-cpp server process is not initialized.")

            if process.poll() is not None:
                raise RuntimeError(
                    "llama-cpp server exited before becoming ready. "
                    "Ensure llama-cpp-python[server] is installed "
                    "(pip install -e '.[llama_cpp]') and server model options are valid."
                )

            try:
                with urlopen(health_url, timeout=1.0) as response:
                    # Any successful HTTP response means the server is up.
                    if 200 <= response.status < 400:
                        return
            except HTTPError as exc:
                # urllib surfaces auth challenges as exceptions even when the server is healthy.
                if exc.code in {401, 403}:
                    return
                # Keep most recent probe failure for timeout diagnostics.
                last_error = exc
            except URLError as exc:
                # Keep most recent probe failure for timeout diagnostics.
                last_error = exc
            except TimeoutError as exc:
                # Slow local startup may time out one probe; continue until deadline.
                last_error = exc

            time.sleep(self.poll_interval_seconds)

        raise RuntimeError(
            f"Timed out waiting for llama-cpp server readiness at {health_url}. Last error: {last_error!r}"
        )

    def close(self) -> None:
        """Stop the managed server process if running.

        Shutdown first attempts a graceful terminate/wait sequence and escalates
        to kill only when termination does not complete within timeout.
        """
        process = self._process
        if process is None:
            return

        if process.poll() is None:
            # Try graceful shutdown first to let llama release resources cleanly.
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Force kill only if graceful shutdown stalls.
                process.kill()
                process.wait(timeout=5.0)

        self._process = None


def create_backend(
    model: str,
    *,
    hf_model_repo_id: str | None = None,
    api_model: str = "local-model",
    host: str = "127.0.0.1",
    port: int = 8001,
    startup_timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 0.25,
    python_executable: str = sys.executable,
    extra_server_args: Sequence[str] = (),
) -> LlamaCppServerBackend:
    """Create a llama-cpp server backend from plain arguments.

    Args:
        model: ``llama_cpp.server`` ``--model`` value.
        hf_model_repo_id: Optional Hugging Face repository id.
        api_model: OpenAI-compatible model alias exposed by the local server.
        host: Server bind host.
        port: Server bind port.
        startup_timeout_seconds: Max startup wait time.
        poll_interval_seconds: Delay between readiness checks.
        python_executable: Python executable used to launch the server.
        extra_server_args: Extra CLI args passed to ``llama_cpp.server``.

    Returns:
        Configured :class:`LlamaCppServerBackend` instance.
    """
    return LlamaCppServerBackend(
        model=model,
        hf_model_repo_id=hf_model_repo_id,
        api_model=api_model,
        host=host,
        port=port,
        startup_timeout_seconds=startup_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        python_executable=python_executable,
        extra_server_args=tuple(extra_server_args),
    )
