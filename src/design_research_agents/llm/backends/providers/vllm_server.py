"""Managed backend for running a local ``vllm`` OpenAI-compatible server."""

from __future__ import annotations

import atexit
import subprocess
import sys
import time
from importlib.util import find_spec
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


class VllmServerBackend:
    """Manage a persistent local ``vllm`` OpenAI-compatible server process."""

    def __init__(
        self,
        *,
        model: str,
        api_model: str,
        host: str = "127.0.0.1",
        port: int = 8002,
        startup_timeout_seconds: float = 90.0,
        poll_interval_seconds: float = 0.5,
        python_executable: str = sys.executable,
        extra_server_args: tuple[str, ...] = (),
    ) -> None:
        """Initialize server lifecycle configuration.

        Args:
            model: Model identifier or local path passed to ``vllm``.
            api_model: Public model name exposed by the OpenAI-compatible API.
            host: Interface used by the managed server.
            port: TCP port used by the managed server.
            startup_timeout_seconds: Maximum startup wait time in seconds.
            poll_interval_seconds: Delay between readiness probes in seconds.
            python_executable: Python executable used to launch ``vllm``.
            extra_server_args: Additional CLI flags forwarded to ``vllm``.

        Raises:
            ValueError: If required string configuration is blank.
        """
        model_value = model.strip()
        if not model_value:
            raise ValueError("model must not be empty.")
        api_model_value = api_model.strip()
        if not api_model_value:
            raise ValueError("api_model must not be empty.")
        host_value = host.strip()
        if not host_value:
            raise ValueError("host must not be empty.")
        if port <= 0:
            raise ValueError("port must be > 0.")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be > 0.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0.")

        self.model = model_value
        self.api_model = api_model_value
        self.host = host_value
        self.port = port
        self.startup_timeout_seconds = startup_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.python_executable = python_executable
        self.extra_server_args = tuple(extra_server_args)
        self._process: subprocess.Popen[str] | None = None
        atexit.register(self.close)

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible API base URL.

        Returns:
            Base URL pointing at the managed server ``/v1`` root.
        """
        return f"http://{self.host}:{self.port}/v1"

    def _build_command(self) -> list[str]:
        """Build the command used to launch ``vllm``.

        Returns:
            CLI command list suitable for ``subprocess.Popen``.
        """
        return [
            self.python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--served-model-name",
            self.api_model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.extra_server_args,
        ]

    def is_running(self) -> bool:
        """Return whether a managed server process is currently alive.

        Returns:
            ``True`` when the managed subprocess exists and has not exited.
        """
        return self._process is not None and self._process.poll() is None

    def _ensure_server_dependency(self) -> None:
        """Validate that ``vllm`` is importable in this Python environment.

        Raises:
            RuntimeError: If the ``vllm`` dependency is missing.
        """
        if find_spec("vllm") is None:
            raise RuntimeError("vLLM dependency is missing. Install with: pip install -e '.[vllm]'")

    def start(self) -> None:
        """Start the managed server process when not already running.

        Raises:
            RuntimeError: If startup fails or readiness probe times out.
        """
        if self.is_running():
            return

        self._ensure_server_dependency()
        self._process = subprocess.Popen(
            self._build_command(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        """Wait until the server responds on ``/v1/models``.

        Raises:
            RuntimeError: If the process exits early or readiness times out.
        """
        deadline = time.monotonic() + self.startup_timeout_seconds
        health_url = f"{self.base_url}/models"
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                raise RuntimeError("vLLM server process is not initialized.")
            if process.poll() is not None:
                raise RuntimeError(
                    "vLLM server exited before becoming ready. "
                    "Ensure vLLM is installed and server arguments are valid."
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

        raise RuntimeError(
            "Timed out waiting for vLLM server readiness at "
            f"{health_url}. Last error: {last_error!r}"
        )

    def close(self) -> None:
        """Stop the managed server process if it is running."""
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
    model: str,
    *,
    api_model: str,
    host: str = "127.0.0.1",
    port: int = 8002,
    startup_timeout_seconds: float = 90.0,
    poll_interval_seconds: float = 0.5,
    python_executable: str = sys.executable,
    extra_server_args: tuple[str, ...] = (),
) -> VllmServerBackend:
    """Construct a ``VllmServerBackend`` using plain arguments.

    Args:
        model: Model identifier or local path passed to ``vllm``.
        api_model: Public model alias exposed by the API server.
        host: Server bind host.
        port: Server bind port.
        startup_timeout_seconds: Maximum startup wait time in seconds.
        poll_interval_seconds: Delay between readiness probes in seconds.
        python_executable: Python executable used to launch ``vllm``.
        extra_server_args: Additional CLI flags forwarded to ``vllm``.

    Returns:
        Configured ``VllmServerBackend`` instance.
    """
    return VllmServerBackend(
        model=model,
        api_model=api_model,
        host=host,
        port=port,
        startup_timeout_seconds=startup_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        python_executable=python_executable,
        extra_server_args=extra_server_args,
    )
