"""Managed backend for running a local ``ollama serve`` process."""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


class OllamaServerBackend:
    """Manage a persistent local Ollama daemon process."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 11434,
        ollama_executable: str = "ollama",
        auto_pull_model: bool = False,
        default_model: str | None = None,
        startup_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        """Initialize Ollama daemon lifecycle configuration.

        Args:
            host: Interface used by the managed Ollama daemon.
            port: TCP port used by the managed Ollama daemon.
            ollama_executable: Executable used to invoke ``ollama`` commands.
            auto_pull_model: Whether to run ``ollama pull`` after startup.
            default_model: Model name used by ``auto_pull_model``.
            startup_timeout_seconds: Maximum startup wait time in seconds.
            poll_interval_seconds: Delay between readiness probes in seconds.

        Raises:
            ValueError: If required string configuration is blank.
        """
        host_value = host.strip()
        if not host_value:
            raise ValueError("host must not be empty.")
        if port <= 0:
            raise ValueError("port must be > 0.")
        executable_value = ollama_executable.strip()
        if not executable_value:
            raise ValueError("ollama_executable must not be empty.")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be > 0.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0.")

        self.host = host_value
        self.port = port
        self.ollama_executable = executable_value
        self.auto_pull_model = auto_pull_model
        self.default_model = default_model.strip() if isinstance(default_model, str) else None
        self.startup_timeout_seconds = startup_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._process: subprocess.Popen[str] | None = None
        atexit.register(self.close)

    @property
    def base_url(self) -> str:
        """Return the managed Ollama HTTP base URL.

        Returns:
            Base URL string for Ollama API requests.
        """
        return f"http://{self.host}:{self.port}"

    def _serve_env(self) -> dict[str, str]:
        """Return environment variables for managed daemon commands.

        Returns:
            Environment mapping used for ``ollama serve`` and ``ollama pull``.
        """
        env = dict(os.environ)
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
        return env

    def _build_command(self) -> list[str]:
        """Build the command used to launch ``ollama serve``.

        Returns:
            CLI command list suitable for ``subprocess.Popen``.
        """
        return [self.ollama_executable, "serve"]

    def is_running(self) -> bool:
        """Return whether a managed daemon process is currently alive.

        Returns:
            ``True`` when the managed subprocess exists and has not exited.
        """
        return self._process is not None and self._process.poll() is None

    def _ensure_server_dependency(self) -> None:
        """Validate that the Ollama executable is available.

        Raises:
            RuntimeError: If the configured executable cannot be found.
        """
        if shutil.which(self.ollama_executable) is None:
            raise RuntimeError(f"Ollama executable '{self.ollama_executable}' was not found in PATH.")

    def start(self) -> None:
        """Start the managed daemon process when not already running.

        Raises:
            RuntimeError: If startup or optional model pull fails.
        """
        if self.is_running():
            return

        self._ensure_server_dependency()
        self._process = subprocess.Popen(
            self._build_command(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            env=self._serve_env(),
        )
        self._wait_until_ready()
        if self.auto_pull_model and self.default_model:
            self._pull_model(self.default_model)

    def _wait_until_ready(self) -> None:
        """Wait until the daemon responds on ``/api/tags``.

        Raises:
            RuntimeError: If the process exits early or readiness times out.
        """
        deadline = time.monotonic() + self.startup_timeout_seconds
        health_url = f"{self.base_url}/api/tags"
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                raise RuntimeError("Ollama server process is not initialized.")
            if process.poll() is not None:
                raise RuntimeError(
                    "Ollama server exited before becoming ready. "
                    "Ensure Ollama is installed and daemon startup is valid."
                )

            try:
                with urlopen(health_url, timeout=1.0) as response:
                    if 200 <= response.status < 400:
                        return
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    return
                last_error = exc
            except (URLError, TimeoutError) as exc:
                last_error = exc

            time.sleep(self.poll_interval_seconds)

        raise RuntimeError(f"Timed out waiting for Ollama server readiness at {health_url}. Last error: {last_error!r}")

    def _pull_model(self, model: str) -> None:
        """Pull one model into the managed Ollama daemon runtime.

        Args:
            model: Model name passed to ``ollama pull``.

        Raises:
            RuntimeError: If model pull command fails.
        """
        completed = subprocess.run(
            [self.ollama_executable, "pull", model],
            env=self._serve_env(),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or "unknown error"
            raise RuntimeError(f"Failed to pull Ollama model '{model}': {details}")

    def close(self) -> None:
        """Stop the managed daemon process if it is running."""
        process = self._process
        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)

        self._process = None


def create_backend(
    *,
    host: str = "127.0.0.1",
    port: int = 11434,
    ollama_executable: str = "ollama",
    auto_pull_model: bool = False,
    default_model: str | None = None,
    startup_timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 0.25,
) -> OllamaServerBackend:
    """Construct an ``OllamaServerBackend`` using plain arguments.

    Args:
        host: Server bind host.
        port: Server bind port.
        ollama_executable: Executable used to invoke ``ollama`` commands.
        auto_pull_model: Whether to run ``ollama pull`` after startup.
        default_model: Model name used by ``auto_pull_model``.
        startup_timeout_seconds: Maximum startup wait time in seconds.
        poll_interval_seconds: Delay between readiness probes in seconds.

    Returns:
        Configured ``OllamaServerBackend`` instance.
    """
    return OllamaServerBackend(
        host=host,
        port=port,
        ollama_executable=ollama_executable,
        auto_pull_model=auto_pull_model,
        default_model=default_model,
        startup_timeout_seconds=startup_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
