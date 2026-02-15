"""Safe runner for lazy tool scripts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

from design_research_agents.contracts.tools import ToolResult
from design_research_agents.tools.policy import ToolPolicy, ToolPolicyError

from .discovery import LazyToolDefinition


class LazyToolRuntimeError(RuntimeError):
    """Raised when lazy script execution output is invalid."""


def run_lazy_tool(
    *,
    tool_name: str,
    definition: LazyToolDefinition,
    input_dict: Mapping[str, object],
    policy: ToolPolicy,
) -> ToolResult:
    """Execute one lazy tool script under policy controls."""
    header = definition.header

    if header.capabilities.network and not policy.config.allow_network:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error="Lazy tool requires network access, but allow_network is disabled.",
        )

    for allowed_command in header.capabilities.commands:
        try:
            policy.validate_command(allowed_command)
        except ToolPolicyError as exc:
            return ToolResult(tool_name=tool_name, ok=False, error=str(exc))

    script_path = Path(definition.path)
    timeout_s = header.timeout_s or policy.config.default_timeout_s

    if script_path.suffix == ".py":
        command = ["python3", str(script_path)]
    elif script_path.suffix == ".sh":
        command = ["/usr/bin/env", "bash", str(script_path)]
    else:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error=f"Unsupported lazy script extension: {script_path.suffix}",
        )

    payload_text = json.dumps(dict(input_dict), ensure_ascii=True)

    try:
        completed = subprocess.run(
            command,
            cwd=str(policy.workspace_root),
            input=payload_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
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
                    f"Lazy tool timed out after {timeout_s}s. stdout={stdout_text[:120]!r} "
                    f"stderr={stderr_text[:120]!r}"
                ),
            },
        )

    stdout_text, stdout_truncated = policy.clamp_output(completed.stdout)
    stderr_text, stderr_truncated = policy.clamp_output(completed.stderr)

    try:
        envelope = _parse_envelope(stdout_text)
    except LazyToolRuntimeError as exc:
        message = str(exc)
        if stderr_text.strip():
            message = f"{message} stderr={stderr_text[:200]!r}"
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error={"type": "LazyOutputError", "message": message},
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
        else {"type": "LazyOutputError", "message": "Invalid error envelope payload."}
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
    stripped = stdout_text.strip()
    if not stripped:
        raise LazyToolRuntimeError(
            "Lazy tool produced empty stdout. Print exactly one JSON object to stdout."
        )

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        snippet = stripped[:200]
        raise LazyToolRuntimeError(
            "Lazy tool stdout must be JSON. Print logs to stderr and print exactly one JSON "
            f"object to stdout. First 200 chars: {snippet!r}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise LazyToolRuntimeError("Lazy tool stdout JSON must be an object.")

    required = {"ok", "result", "artifacts", "warnings"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise LazyToolRuntimeError(f"Lazy tool JSON envelope is missing keys: {', '.join(missing)}")

    return payload


__all__ = ["LazyToolRuntimeError", "run_lazy_tool"]
