"""Tool source for explicitly configured script-backed tools."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolResult,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._config import ScriptToolConfig
from design_research_agents.tools._policy import ToolPolicy, ToolPolicyError


class ScriptToolRuntimeError(RuntimeError):
    """Raised when script-tool execution output is invalid."""


@dataclass(slots=True, frozen=True, kw_only=True)
class _ScriptToolBinding:
    """Resolved binding for one configured script-backed tool."""

    canonical_name: str
    """Canonical public tool name exposed by the runtime."""
    config: ScriptToolConfig
    """Script configuration that drives invocation and policy checks."""


class ScriptToolSource:
    """Execute explicitly declared script tools."""

    source_id = "script"

    def __init__(self, *, script_tools: tuple[ScriptToolConfig, ...], policy: ToolPolicy) -> None:
        """Initialize script tool bindings with policy controls.

        Args:
            script_tools: Explicit script tool definitions to register.
            policy: Tool policy used for command validation and sandbox defaults.
        """
        self._policy = policy
        self._bindings: dict[str, _ScriptToolBinding] = {}
        for script_tool in script_tools:
            canonical_name = _canonical_script_tool_name(script_tool.name)
            # Last write wins for duplicate names; callers should treat names as unique identifiers.
            self._bindings[canonical_name] = _ScriptToolBinding(
                canonical_name=canonical_name,
                config=script_tool,
            )

    def list_tools(self) -> Sequence[ToolSpec]:
        """List script-backed tool specs.

        Returns:
            Result produced by this call.
        """
        specs: list[ToolSpec] = []
        for canonical_name, binding in sorted(self._bindings.items()):
            config = binding.config
            specs.append(
                ToolSpec(
                    name=canonical_name,
                    description=config.description,
                    input_schema=dict(config.input_schema),
                    output_schema=dict(config.output_schema),
                    permissions=config.permissions,
                    metadata=ToolMetadata(
                        source="script",
                        side_effects=ToolSideEffects(
                            filesystem_read=config.filesystem_read,
                            filesystem_write=config.filesystem_write,
                            network=config.network,
                            commands=config.commands,
                        ),
                        timeout_s=config.timeout_s,
                        max_output_bytes=self._policy.config.default_max_output_bytes,
                        risky=config.risky,
                    ),
                )
            )
        return tuple(specs)

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one configured script tool by canonical name.

        Args:
            tool_name: Value supplied for ``tool_name``.
            input_dict: Value supplied for ``input_dict``.
            request_id: Value supplied for ``request_id``.
            dependencies: Value supplied for ``dependencies``.

        Returns:
            Result produced by this call.
        """
        del request_id, dependencies
        binding = self._bindings.get(tool_name)
        if binding is None:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"Unknown script tool '{tool_name}'.",
            )
        return _run_script_tool(
            tool_name=tool_name,
            config=binding.config,
            input_dict=input_dict,
            policy=self._policy,
        )


def _canonical_script_tool_name(raw_name: str) -> str:
    """Normalize a configured script tool name into the script namespace.

    Args:
        raw_name: User-declared tool name from configuration.

    Returns:
        Canonical tool name prefixed with ``script::``.
    """
    normalized = raw_name.strip()
    if normalized.startswith("script::"):
        return normalized
    # Prefix names to avoid collisions with core/MCP tool namespaces.
    return f"script::{normalized}"


def _run_script_tool(
    *,
    tool_name: str,
    config: ScriptToolConfig,
    input_dict: Mapping[str, object],
    policy: ToolPolicy,
) -> ToolResult:
    """Execute one configured script tool and normalize its response.

    Args:
        tool_name: Canonical tool name exposed by the runtime.
        config: Script tool configuration that defines the executable and policy.
        input_dict: JSON-serializable tool-input payload.
        policy: Tool policy used to validate commands and clamp output.

    Returns:
        Tool result representing success, failure, or envelope-parse errors.
    """
    if config.network and not policy.config.allow_network:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error="Script tool requires network access, but allow_network is disabled.",
        )

    for allowed_command in config.commands:
        try:
            policy.validate_command(allowed_command)
        except ToolPolicyError as exc:
            return ToolResult(tool_name=tool_name, ok=False, error=str(exc))

    script_path = Path(config.path)
    if script_path.suffix == ".py":
        # Execute Python scripts via python3 for predictable interpreter selection in tool sandboxes.
        command = ["python3", str(script_path)]
    elif script_path.suffix == ".sh":
        command = ["/usr/bin/env", "bash", str(script_path)]
    else:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error=f"Unsupported script extension: {script_path.suffix}",
        )

    payload_text = json.dumps(dict(input_dict), ensure_ascii=True)

    try:
        completed = subprocess.run(
            command,
            cwd=str(policy.workspace_root),
            input=payload_text,
            capture_output=True,
            text=True,
            timeout=config.timeout_s,
            check=False,
            env=policy.sanitize_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error={
                "type": "TimeoutError",
                "message": (
                    f"Script tool timed out after {config.timeout_s}s. "
                    f"stdout={stdout_text[:120]!r} "
                    f"stderr={stderr_text[:120]!r}"
                ),
            },
        )

    stdout_text, stdout_truncated = policy.clamp_output(completed.stdout)
    stderr_text, stderr_truncated = policy.clamp_output(completed.stderr)

    try:
        envelope = _parse_envelope(stdout_text)
    except ScriptToolRuntimeError as exc:
        # Keep parse failures structured so agent loops can report tool errors uniformly.
        message = str(exc)
        if stderr_text.strip():
            message = f"{message} stderr={stderr_text[:200]!r}"
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error={"type": "ScriptOutputError", "message": message},
            metadata={
                "returncode": completed.returncode,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
        )

    artifacts_raw = envelope.get("artifacts", ())
    warnings_raw = envelope.get("warnings", ())
    error_raw = envelope.get("error")
    artifacts = artifacts_raw if isinstance(artifacts_raw, (list, tuple)) else ()
    warnings = warnings_raw if isinstance(warnings_raw, (list, tuple)) else ()
    error = (
        error_raw
        if isinstance(error_raw, (str, Mapping)) or error_raw is None
        else {"type": "ScriptOutputError", "message": "Invalid error envelope payload."}
    )

    result = ToolResult(
        tool_name=tool_name,
        ok=bool(envelope.get("ok", False)),
        result=envelope.get("result", {}),
        artifacts=artifacts,
        warnings=warnings,
        error=error,
        metadata={
            "returncode": completed.returncode,
            "stderr": stderr_text,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "script_path": str(script_path),
        },
    )

    try:
        policy.validate_result_artifacts(result)
    except ToolPolicyError as exc:
        return ToolResult(tool_name=tool_name, ok=False, error=str(exc))

    return result


def _parse_envelope(stdout_text: str) -> Mapping[str, object]:
    """Parse envelope.

    Args:
        stdout_text: Value supplied for ``stdout_text``.

    Returns:
        Result produced by this call.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    stripped = stdout_text.strip()
    if not stripped:
        raise ScriptToolRuntimeError("Script tool produced empty stdout. Print exactly one JSON object to stdout.")

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        snippet = stripped[:200]
        raise ScriptToolRuntimeError(
            "Script tool stdout must be JSON. Print logs to stderr and print exactly one JSON "
            f"object to stdout. First 200 chars: {snippet!r}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ScriptToolRuntimeError("Script tool stdout JSON must be an object.")

    required = {"ok", "result", "artifacts", "warnings"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ScriptToolRuntimeError(f"Script tool JSON envelope is missing keys: {', '.join(missing)}")

    return payload


__all__ = ["ScriptToolRuntimeError", "ScriptToolSource"]
