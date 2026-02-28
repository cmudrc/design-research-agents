"""Policy enforcement helpers for tool execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from design_research_agents._contracts._tools import ToolArtifact, ToolResult, ToolSpec


class ToolPolicyError(RuntimeError):
    """Raised when a tool invocation violates runtime policy."""


@dataclass(slots=True, frozen=True, kw_only=True)
class ToolPolicyConfig:
    """Runtime guardrail settings used by core, MCP, and script tools."""

    workspace_root: str = "."
    """Workspace root used as the trust boundary for path validation."""
    artifacts_dir: str = "artifacts"
    """Workspace-relative directory allowed for writes by default."""
    allow_writes_outside_artifacts: bool = False
    """Whether writes outside ``artifacts/`` are permitted."""
    allow_network: bool = False
    """Whether tools may perform network I/O."""
    allowed_commands: tuple[str, ...] = (
        "git",
        "rg",
        "python",
        "python3",
        "uv",
        "ruff",
        "pytest",
    )
    """Command allowlist enforced for subprocess-based tools."""
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
    """Environment variables preserved for subprocess execution."""
    default_timeout_s: int = 30
    """Default subprocess timeout in seconds."""
    default_max_output_bytes: int = 65_536
    """Default maximum captured subprocess output size in bytes."""


class ToolPolicy:
    """Policy engine for validating side effects and runtime boundaries."""

    def __init__(self, config: ToolPolicyConfig) -> None:
        """Initialize policy with resolved workspace and artifacts roots.

        Args:
            config: Immutable runtime policy configuration.
        """
        self._config = config
        # Resolve once at construction time so all checks use canonical absolute paths.
        self._workspace_root = Path(config.workspace_root).expanduser().resolve()
        self._artifacts_root = (self._workspace_root / config.artifacts_dir).resolve()

    @property
    def workspace_root(self) -> Path:
        """Return resolved workspace root directory.

        Returns:
            Absolute workspace root path.
        """
        return self._workspace_root

    @property
    def artifacts_root(self) -> Path:
        """Return resolved artifacts output directory.

        Returns:
            Absolute artifacts root path.
        """
        return self._artifacts_root

    @property
    def config(self) -> ToolPolicyConfig:
        """Return immutable policy configuration.

        Returns:
            Policy configuration object backing this policy engine.
        """
        return self._config

    def validate_tool_spec(self, spec: ToolSpec) -> None:
        """Validate a tool can run under current global policy settings.

        Args:
            spec: Tool specification to validate against global policy.

        Raises:
            ToolPolicyError: If the tool requires disallowed side effects.
        """
        side_effects = spec.metadata.side_effects
        # Network access is guarded globally regardless of per-tool preference.
        if side_effects.network and not self._config.allow_network:
            raise ToolPolicyError(f"Tool '{spec.name}' requires network access, but allow_network is disabled.")

    def validate_command(self, command: str) -> None:
        """Reject commands not present in policy allowlist.

        Args:
            command: Command name requested by a subprocess-based tool.

        Raises:
            ToolPolicyError: If the command is blank or not allowlisted.
        """
        normalized = command.strip()
        if not normalized:
            raise ToolPolicyError("Command cannot be empty.")
        allowed = set(self._config.allowed_commands)
        if normalized not in allowed:
            raise ToolPolicyError(f"Command '{normalized}' is not in the allowed_commands policy list.")

    def resolve_read_path(self, path: str | Path) -> Path:
        """Resolve and validate a readable path inside the workspace root.

        Args:
            path: Candidate path that must stay within the workspace.

        Returns:
            Absolute readable path inside the workspace root.

        Raises:
            ToolPolicyError: If the path escapes the workspace or does not exist.
        """
        candidate = self._resolve_workspace_path(path)
        if not candidate.exists():
            raise ToolPolicyError(f"Path does not exist: {candidate}")
        return candidate

    def resolve_write_path(self, path: str | Path) -> Path:
        """Resolve and validate a writable path under policy rules.

        Args:
            path: Candidate path that must satisfy write policy.

        Returns:
            Absolute writable path permitted by the current policy.

        Raises:
            ToolPolicyError: If the path escapes the workspace or write boundary.
        """
        candidate = self._resolve_workspace_path(path)
        if self._config.allow_writes_outside_artifacts:
            return candidate
        # Default safety policy constrains writes to artifacts/ for reproducible side effects.
        if not self._is_relative_to(candidate, self._artifacts_root):
            raise ToolPolicyError(
                "Writes are restricted to the artifacts directory. "
                f"Got '{candidate}', expected under '{self._artifacts_root}'."
            )
        return candidate

    def clamp_output(self, text: str, max_output_bytes: int | None = None) -> tuple[str, bool]:
        """Truncate UTF-8 text to configured output byte limits.

        Args:
            text: Captured text to clamp.
            max_output_bytes: Optional byte limit override.

        Returns:
            Tuple of ``(clipped_text, was_truncated)``.
        """
        limit = max_output_bytes or self._config.default_max_output_bytes
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= limit:
            return text, False
        # Decode with ignore to avoid cutting through multibyte UTF-8 boundaries.
        clipped = encoded[:limit].decode("utf-8", errors="ignore")
        return clipped, True

    def validate_result_artifacts(self, result: ToolResult) -> None:
        """Ensure artifact paths obey write policy when applicable.

        Args:
            result: Tool result whose artifact paths should be validated.

        Raises:
            ToolPolicyError: If an artifact escapes the configured write boundary.
        """
        if self._config.allow_writes_outside_artifacts:
            return
        for artifact in result.artifacts:
            if not isinstance(artifact, ToolArtifact):
                continue
            artifact_path = self._resolve_workspace_path(artifact.path)
            if not self._is_relative_to(artifact_path, self._artifacts_root):
                raise ToolPolicyError(
                    f"Tool '{result.tool_name}' produced artifact outside artifacts root: {artifact.path}"
                )

    def sanitize_subprocess_env(
        self,
        *,
        allowlist: tuple[str, ...] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Return allowlisted environment variables for subprocesses.

        Args:
            allowlist: Optional environment allowlist override.
            extra_env: Optional explicit environment values to merge in.

        Returns:
            Sanitized environment mapping safe for subprocess use.
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
        """Resolve one path and enforce the workspace-root boundary.

        Args:
            path: Candidate path to resolve.

        Returns:
            Absolute resolved path inside the workspace root.

        Raises:
            ToolPolicyError: If the resolved path escapes the workspace root.
        """
        raw_path = Path(path).expanduser()
        if not raw_path.is_absolute():
            raw_path = self._workspace_root / raw_path
        resolved = raw_path.resolve()
        if not self._is_relative_to(resolved, self._workspace_root):
            raise ToolPolicyError(f"Path '{resolved}' is outside workspace root '{self._workspace_root}'.")
        return resolved

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        """Return whether ``path`` is nested under ``root``.

        Args:
            path: Candidate path to test.
            root: Boundary root path.

        Returns:
            ``True`` when ``path`` is inside ``root``.
        """
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


__all__ = ["ToolPolicy", "ToolPolicyConfig", "ToolPolicyError"]
