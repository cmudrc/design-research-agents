from __future__ import annotations

import pytest

from design_research_agents.tools import _config as cfg


def test_parse_primitives_and_env() -> None:
    assert cfg._parse_str(" value ") == "value"
    assert cfg._parse_str("   ") is None
    assert cfg._parse_str(1) is None

    assert cfg._parse_str_list([" a ", "", "b"]) == ("a", "b")
    assert cfg._parse_str_list(None) == ()
    with pytest.raises(ValueError, match="list of strings"):
        cfg._parse_str_list("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="list of strings"):
        cfg._parse_str_list(["ok", 1])  # type: ignore[list-item]

    assert cfg._parse_int(None, default=5) == 5
    assert cfg._parse_int(9, default=1) == 9
    with pytest.raises(ValueError, match="integer"):
        cfg._parse_int(True, default=1)

    assert cfg._parse_env(None) == {}
    assert cfg._parse_env({" A ": 3, "B": "x"}) == {"A": "3", "B": "x"}
    with pytest.raises(ValueError, match="env mapping"):
        cfg._parse_env("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        cfg._parse_env({"": "x"})


def test_parse_core_and_script_configs() -> None:
    core = cfg._parse_core_config(
        {
            "enabled": False,
            "allow_network": True,
            "allow_writes_outside_artifacts": True,
            "allowed_commands": ["git", "python3"],
            "artifacts_dir": "out",
            "workspace_root": " /workspace ",
        }
    )
    assert core.enabled is False
    assert core.allow_network is True
    assert core.allow_writes_outside_artifacts is True
    assert core.allowed_commands == ("git", "python3")
    assert core.artifacts_dir == "out"
    assert core.workspace_root == "/workspace"

    assert cfg._parse_core_config("bad") == cfg.CoreToolsConfig()

    script = cfg._parse_script_config(
        {
            "enabled": True,
            "tools": [
                {
                    "name": "quick",
                    "path": " /tmp/quick.py ",
                    "description": " quick tool ",
                    "filesystem_read": True,
                    "filesystem_write": True,
                    "network": False,
                    "commands": ["uv", "pytest"],
                    "timeout_s": 99,
                    "permissions": ["tool:quick"],
                }
            ],
        }
    )
    assert script.enabled is True
    assert len(script.tools) == 1
    tool = script.tools[0]
    assert tool.name == "quick"
    assert tool.path == "/tmp/quick.py"
    assert tool.description == "quick tool"
    assert tool.filesystem_read is True
    assert tool.filesystem_write is True
    assert tool.network is False
    assert tool.commands == ("uv", "pytest")
    assert tool.timeout_s == 99
    assert tool.permissions == ("tool:quick",)

    assert cfg._parse_script_config("bad") == cfg.ScriptToolsConfig()


def test_parse_mcp_config_validation_and_defaults() -> None:
    assert cfg._parse_mcp_config("bad") == cfg.McpConfig()

    with pytest.raises(ValueError, match="must be a mapping"):
        cfg._parse_mcp_config({"servers": [123]})

    with pytest.raises(ValueError, match="id is required"):
        cfg._parse_mcp_config({"servers": [{"command": ["cmd"]}]})

    with pytest.raises(ValueError, match="not supported"):
        cfg._parse_mcp_config({"servers": [{"id": "a", "type": "http", "command": ["cmd"]}]})

    with pytest.raises(ValueError, match="non-empty string list"):
        cfg._parse_mcp_config({"servers": [{"id": "a", "command": []}]})

    parsed = cfg._parse_mcp_config(
        {
            "enabled": True,
            "servers": [
                {
                    "id": "alpha",
                    "type": "stdio",
                    "command": ["python", "server.py"],
                    "timeout_s": 11,
                    "env_allowlist": ["PATH", "HOME"],
                    "env": {"TOKEN": 123},
                }
            ],
        }
    )
    assert parsed.enabled is True
    assert len(parsed.servers) == 1
    server = parsed.servers[0]
    assert server.id == "alpha"
    assert server.type == "stdio"
    assert server.command == ("python", "server.py")
    assert server.timeout_s == 11
    assert server.env_allowlist == ("PATH", "HOME")
    assert server.env == {"TOKEN": "123"}


def test_load_tool_runtime_config_from_json(tmp_path) -> None:
    config_file = tmp_path / "tools.json"
    config_file.write_text(
        (
            "{\n"
            '  "core_tools": {"enabled": false, "allow_network": true},\n'
            '  "mcp": {\n'
            '    "enabled": true,\n'
            '    "servers": [{"id": "alpha", "command": ["echo", "hello"]}]\n'
            "  },\n"
            '  "script_tools": {\n'
            '    "enabled": true,\n'
            '    "tools": [\n'
            '      {"name": "quick", "path": "/tmp/quick.py", "description": "quick"}\n'
            "    ]\n"
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    loaded = cfg.load_tool_runtime_config(str(config_file))
    assert loaded.core_tools.enabled is False
    assert loaded.core_tools.allow_network is True
    assert loaded.mcp.enabled is True
    assert loaded.mcp.servers[0].id == "alpha"
    assert loaded.script_tools.enabled is True
    assert loaded.script_tools.tools[0].name == "quick"

    legacy_file = tmp_path / "legacy.json"
    legacy_file.write_text('{"lazy_tools": {"enabled": true}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="script_tools"):
        cfg.load_tool_runtime_config(str(legacy_file))

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text('["not-a-mapping"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        cfg.load_tool_runtime_config(str(invalid_file))

    malformed_file = tmp_path / "malformed.json"
    malformed_file.write_text("{bad-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        cfg.load_tool_runtime_config(str(malformed_file))
