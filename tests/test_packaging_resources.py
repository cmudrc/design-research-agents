from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_RESOURCES = (
    "design_research_agents/_schemas/execution_result.schema.json",
    "design_research_agents/_schemas/tool_result.schema.json",
    "design_research_agents/_schemas/tool_spec.schema.json",
)

PROMPT_RESOURCES = (
    "design_research_agents/_prompts/code_action_step_system.md",
    "design_research_agents/_prompts/code_action_step_user_plan.md",
    "design_research_agents/_prompts/multi_step_continue_system.md",
    "design_research_agents/_prompts/multi_step_continue_user.md",
    "design_research_agents/_prompts/multi_step_direct_controller_system.md",
    "design_research_agents/_prompts/multi_step_direct_controller_user.md",
    "design_research_agents/_prompts/multi_step_json_step_user.md",
    "design_research_agents/_prompts/multi_step_step_user.md",
    "design_research_agents/_prompts/tool_calling_system.md",
    "design_research_agents/_prompts/tool_calling_user_select_tool.md",
)


def _build_wheel(tmp_path: Path) -> Path:
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(wheel_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Failed to build wheel.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    wheels = sorted(wheel_dir.glob("design_research_agents-*.whl"))
    assert wheels, f"No wheel generated in {wheel_dir}."
    return wheels[0]


def test_wheel_includes_prompt_and_schema_resources(tmp_path: Path) -> None:
    wheel_path = _build_wheel(tmp_path)
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())

    missing = sorted(resource for resource in (*SCHEMA_RESOURCES, *PROMPT_RESOURCES) if resource not in names)
    assert not missing, f"Wheel is missing packaged resources: {missing}"


def test_installed_wheel_loads_prompt_and_schema_resources(tmp_path: Path) -> None:
    wheel_path = _build_wheel(tmp_path)
    install_dir = tmp_path / "install"
    install_dir.mkdir(parents=True, exist_ok=True)

    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, (
        f"Failed to install built wheel.\nstdout:\n{install.stdout}\nstderr:\n{install.stderr}"
    )

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    install_pythonpath = str(install_dir)
    env["PYTHONPATH"] = (
        f"{install_pythonpath}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else install_pythonpath
    )

    runner_code = """
import json
from design_research_agents._prompts import PROMPT_NAMES, load_prompt
from design_research_agents._schemas import load_schema

schema = load_schema("tool_spec")
prompt = load_prompt(PROMPT_NAMES[0])
print(json.dumps({"schema_type": schema.get("type"), "prompt_len": len(prompt)}))
"""
    probe = subprocess.run(
        [sys.executable, "-c", runner_code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, (
        f"Installed wheel could not load packaged resources.\nstdout:\n{probe.stdout}\nstderr:\n{probe.stderr}"
    )
    payload = json.loads(probe.stdout.strip())
    assert payload["schema_type"] == "object"
    assert isinstance(payload["prompt_len"], int)
    assert payload["prompt_len"] > 0
