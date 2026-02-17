"""Policy enforcement helpers for tool execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from design_research_agents.contracts.tools import ToolArtifact, ToolResult, ToolSpec


class ToolPolicyError(RuntimeError):
    """Raised when a tool invocation violates runtime policy."""


@dataclass(slots=True, frozen=True)
class ToolPolicyConfig:
    """Runtime guardrail settings used by core, MCP, and script tools."""

    workspace_root: str = "."
    """Field value for ``workspace_root``."""
    artifacts_dir: str = "artifacts"
    """Field value for ``artifacts_dir``."""
    allow_writes_outside_artifacts: bool = False
    """Field value for ``allow_writes_outside_artifacts``."""
    allow_network: bool = False
    """Field value for ``allow_network``."""
    allowed_commands: tuple[str, ...] = (
        "git",
        "rg",
        "python",
        "python3",
        "uv",
        "ruff",
        "pytest",
    )
    """Field value for ``allowed_commands``."""
    env_allowlist: tuple[str, ...] = (
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    )
    """Field value for ``env_allowlist``."""
    default_timeout_s: int = 30
    """Field value for ``default_timeout_s``."""
    default_max_output_bytes: int = 65_536
    """Field value for ``default_max_output_bytes``."""


class ToolPolicy:
    """Policy engine for validating side effects and runtime boundaries."""

    def __init__(self, config: ToolPolicyConfig) -> None:
        """Initialize policy with resolved workspace and artifacts roots.

        Args:
            config: Parameter value.
        """
        self._config = config
        self._workspace_root = Path(config.workspace_root).expanduser().resolve()
        self._artifacts_root = (self._workspace_root / config.artifacts_dir).resolve()

    @property
    def workspace_root(self) -> Path:
        """Return resolved workspace root directory.

        Returns:
            The resulting value.
        """
        return self._workspace_root

    @property
    def artifacts_root(self) -> Path:
        """Return resolved artifacts output directory.

        Returns:
            The resulting value.
        """
        return self._artifacts_root

    @property
    def config(self) -> ToolPolicyConfig:
        """Return immutable policy configuration.

        Returns:
            The resulting value.
        """
        return self._config

    def validate_tool_spec(self, spec: ToolSpec) -> None:
        """Validate a tool can run under current global policy settings.

        Args:
            spec: Parameter value.

        Raises:
            Exception: Raised when execution fails.
        """
        side_effects = spec.metadata.side_effects
        if side_effects.network and not self._config.allow_network:
            raise ToolPolicyError(
                f"Tool '{spec.name}' requires network access, but allow_network is disabled."
            )

    def validate_command(self, command: str) -> None:
        """Reject commands not present in policy allowlist.

        Args:
            command: Parameter value.

        Raises:
            Exception: Raised when execution fails.
        """
        normalized = command.strip()
        if not normalized:
            raise ToolPolicyError("Command cannot be empty.")
        allowed = set(self._config.allowed_commands)
        if normalized not in allowed:
            raise ToolPolicyError(
                f"Command '{normalized}' is not in the allowed_commands policy list."
            )

    def resolve_read_path(self, path: str | Path) -> Path:
        """Resolve and validate a readable path inside the workspace root.

        Args:
            path: Parameter value.

        Returns:
            The resulting value.

        Raises:
            Exception: Raised when execution fails.
        """
        candidate = self._resolve_workspace_path(path)
        if not candidate.exists():
            raise ToolPolicyError(f"Path does not exist: {candidate}")
        return candidate

    def resolve_write_path(self, path: str | Path) -> Path:
        """Resolve and validate a writable path under policy rules.

        Args:
            path: Parameter value.

        Returns:
            The resulting value.

        Raises:
            Exception: Raised when execution fails.
        """
        candidate = self._resolve_workspace_path(path)
        if self._config.allow_writes_outside_artifacts:
            return candidate
        if not self._is_relative_to(candidate, self._artifacts_root):
            raise ToolPolicyError(
                "Writes are restricted to the artifacts directory. "
                f"Got '{candidate}', expected under '{self._artifacts_root}'."
            )
        return candidate

    def clamp_output(self, text: str, max_output_bytes: int | None = None) -> tuple[str, bool]:
        """Truncate UTF-8 text to configured output byte limits.

        Args:
            text: Parameter value.
            max_output_bytes: Parameter value.

        Returns:
            The resulting value.
        """
        limit = max_output_bytes or self._config.default_max_output_bytes
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= limit:
            return text, False
        clipped = encoded[:limit].decode("utf-8", errors="ignore")
        return clipped, True

    def validate_result_artifacts(self, result: ToolResult) -> None:
        """Ensure artifact paths obey write policy when applicable.

        Args:
            result: Parameter value.

        Raises:
            Exception: Raised when execution fails.
        """
        if self._config.allow_writes_outside_artifacts:
            return
        for artifact in result.artifacts:
            if not isinstance(artifact, ToolArtifact):
                continue
            artifact_path = self._resolve_workspace_path(artifact.path)
            if not self._is_relative_to(artifact_path, self._artifacts_root):
                raise ToolPolicyError(
                    f"Tool '{result.tool_name}' produced artifact outside artifacts root: "
                    f"{artifact.path}"
                )

    def sanitize_subprocess_env(
        self,
        *,
        allowlist: tuple[str, ...] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Return allowlisted environment variables for subprocesses.

        Args:
            allowlist: Parameter value.
            extra_env: Parameter value.

        Returns:
            The resulting value.
        """
        selected = allowlist if allowlist is not None else self._config.env_allowlist
        env: dict[str, str] = {}
        for key in selected:
            if key in os.environ:
                env[key] = os.environ[key]
        if extra_env:
            for key, value in extra_env.items():
                if key in selected:
                    env[key] = value
        return env

    def _resolve_workspace_path(self, path: str | Path) -> Path:
        """Run resolve workspace path.

        Args:
            path: Parameter value.

        Returns:
            The resulting value.

        Raises:
            Exception: Raised when execution fails.
        """
        raw_path = Path(path).expanduser()
        if not raw_path.is_absolute():
            raw_path = self._workspace_root / raw_path
        resolved = raw_path.resolve()
        if not self._is_relative_to(resolved, self._workspace_root):
            raise ToolPolicyError(
                f"Path '{resolved}' is outside workspace root '{self._workspace_root}'."
            )
        return resolved

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        """Run is relative to.

        Args:
            path: Parameter value.
            root: Parameter value.

        Returns:
            The resulting value.
        """
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


__all__ = ["ToolPolicy", "ToolPolicyConfig", "ToolPolicyError"]
