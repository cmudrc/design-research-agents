from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
EXAMPLE_MONKEYPATCH_ROOT = REPO_ROOT / "tests" / "example_monkeypatch"
pytestmark = pytest.mark.examples_full

_SUMMARY_REQUIRED_KEYS = (
    "success",
    "final_output",
    "terminated_reason",
    "error",
    "trace",
)

_EXECUTION_RESULT_EXAMPLES = {
    "examples/agents/direct_llm_call.py",
    "examples/agents/direct_llm_compiled_execution.py",
    "examples/agents/multi_step_direct_llm_agent.py",
    "examples/agents/multi_step_json_tool_calling_agent.py",
    "examples/agents/multi_step_code_tool_calling_agent.py",
    "examples/agents/multi_step_json_with_memory.py",
    "examples/workflow/workflow_runtime.py",
    "examples/workflow/workflow_runtime_loop_step.py",
    "examples/workflow/workflow_prompt_mode.py",
    "examples/workflow/workflow_schema_mode.py",
    "examples/workflow/workflow_model_step_design_tradeoff.py",
    "examples/workflow/workflow_delegate_and_memory_steps.py",
    "examples/patterns/router_delegate.py",
    "examples/patterns/two_speaker_conversation.py",
    "examples/patterns/debate_pattern.py",
    "examples/patterns/plan_execute.py",
    "examples/patterns/propose_critic.py",
    "examples/patterns/rag.py",
    "examples/patterns/beam_search.py",
    "examples/patterns/coordination_patterns.py",
    "examples/optimization/multi_step_json_tool_calling_1d_optimization.py",
}

_NESTED_SUMMARY_EXAMPLES: dict[str, tuple[str, ...]] = {
    "examples/workflow/workflow_prompt_mode.py": ("agent_branch_run", "template_branch_run"),
    "examples/workflow/workflow_schema_mode.py": ("strict_run", "relaxed_run"),
    "examples/patterns/coordination_patterns.py": ("blackboard", "round_based_coordination"),
}


def _assert_summary_envelope(payload: object, *, context: str) -> None:
    assert isinstance(payload, dict), f"{context} should output one JSON object."
    missing = sorted(set(_SUMMARY_REQUIRED_KEYS).difference(payload.keys()))
    assert not missing, f"{context} summary is missing keys: {missing}"
    assert isinstance(payload.get("success"), bool), f"{context} summary.success must be bool."
    terminated_reason = payload.get("terminated_reason")
    assert terminated_reason is None or isinstance(terminated_reason, str), (
        f"{context} summary.terminated_reason must be string-or-null."
    )
    assert isinstance(payload.get("trace"), dict), f"{context} summary.trace must be an object."


def _discover_non_streaming_examples() -> tuple[str, ...]:
    collected: list[str] = []
    for path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        parts = relative.parts
        if "__pycache__" in parts:
            continue
        if "streaming" in parts:
            continue
        if path.name.startswith("_"):
            continue
        collected.append(str(relative))
    return tuple(collected)


NON_STREAMING_EXAMPLES = _discover_non_streaming_examples()


@pytest.mark.parametrize("example_relpath", sorted(_EXECUTION_RESULT_EXAMPLES))
def test_execution_result_examples_build_summary_without_post_mutation(example_relpath: str) -> None:
    source = (REPO_ROOT / example_relpath).read_text(encoding="utf-8")
    assert 'summary["' not in source, f"{example_relpath} should not mutate summary with item assignment."
    assert "summary['" not in source, f"{example_relpath} should not mutate summary with item assignment."
    assert "summary.update(" not in source, f"{example_relpath} should not mutate summary with update()."
    assert "trace_provider=" not in source, f"{example_relpath} should not pass trace_provider to summary()."
    assert re.search(r"summary\([^)]*request_id=", source, flags=re.DOTALL) is None, (
        f"{example_relpath} should not pass request_id to summary()."
    )
    assert re.search(r"summary\([^)]*details=", source, flags=re.DOTALL) is None, (
        f"{example_relpath} should not pass details to summary()."
    )


@pytest.mark.parametrize("example_relpath", NON_STREAMING_EXAMPLES)
def test_non_streaming_example_runs(
    example_relpath: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRA_EXAMPLE_LLM_MODE", "deterministic")
    example_path = REPO_ROOT / example_relpath
    env = dict(os.environ)
    env["DRA_EXAMPLE_ID"] = example_relpath.replace("\\", "/")
    env["DRA_EXAMPLE_MCP_COMMAND"] = f"{sys.executable} -m design_research_agents._mcp_server"
    existing_pythonpath = env.get("PYTHONPATH")
    test_paths = f"{EXAMPLE_MONKEYPATCH_ROOT}{os.pathsep}src"
    env["PYTHONPATH"] = f"{test_paths}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else test_paths

    completed = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{example_relpath} failed with exit code {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert completed.stdout.strip(), f"{example_relpath} produced empty stdout."

    if example_relpath not in _EXECUTION_RESULT_EXAMPLES:
        return

    parsed = json.loads(completed.stdout)
    nested_keys = _NESTED_SUMMARY_EXAMPLES.get(example_relpath)
    if nested_keys is None:
        _assert_summary_envelope(parsed, context=example_relpath)
        return

    assert isinstance(parsed, dict), f"{example_relpath} should output one JSON object."
    for nested_key in nested_keys:
        assert nested_key in parsed, f"{example_relpath} is missing nested summary '{nested_key}'."
        _assert_summary_envelope(parsed[nested_key], context=f"{example_relpath}:{nested_key}")
