from __future__ import annotations

import pytest

from design_research_agents.tools import config as cfg


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


def test_parse_core_and_lazy_configs() -> None:
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

    lazy = cfg._parse_lazy_config(
        {
            "enabled": True,
            "search_paths": ["tools", " ~/.dra/tools "],
            "allow_network": True,
            "allow_writes_outside_artifacts": True,
            "allowed_commands": ["uv", "pytest"],
            "timeout_s_default": 99,
        }
    )
    assert lazy.enabled is True
    assert lazy.search_paths == ("tools", "~/.dra/tools")
    assert lazy.allow_network is True
    assert lazy.allow_writes_outside_artifacts is True
    assert lazy.allowed_commands == ("uv", "pytest")
    assert lazy.timeout_s_default == 99

    assert cfg._parse_lazy_config("bad") == cfg.LazyToolsConfig()


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


def test_load_tool_runtime_config_from_yaml(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "tools.yaml"
    config_file.write_text(
        "\n".join(
            [
                "core_tools:",
                "  enabled: false",
                "  allow_network: true",
                "mcp:",
                "  enabled: true",
                "  servers:",
                "    - id: alpha",
                '      command: ["echo", "hello"]',
                "lazy_tools:",
                "  enabled: true",
                "  timeout_s_default: 50",
            ]
        ),
        encoding="utf-8",
    )

    loaded = cfg.load_tool_runtime_config(str(config_file))
    assert loaded.core_tools.enabled is False
    assert loaded.core_tools.allow_network is True
    assert loaded.mcp.enabled is True
    assert loaded.mcp.servers[0].id == "alpha"
    assert loaded.lazy_tools.enabled is True
    assert loaded.lazy_tools.timeout_s_default == 50

    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        cfg.load_tool_runtime_config(str(invalid_file))

    monkeypatch.setattr(cfg, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML"):
        cfg.load_tool_runtime_config(str(config_file))
