from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from design_research_agents import SkillsConfig
from design_research_agents._contracts._llm import (
    LLMChatParams,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents._skills import SkillsToolRuntimeAdapter, discover_skills, resolve_skills_context
from design_research_agents._skills._parser import parse_skill_file
from design_research_agents.agent import DirectLLMCall, MultiStepAgent
from design_research_agents.patterns import RouterDelegatePattern, TwoSpeakerConversationPattern
from design_research_agents.tools import Toolbox
from tests.helpers.workflow_stubs import StaticMarkerAgent


class _CaptureSequenceLLMClient:
    def __init__(self, *, response_texts: Sequence[str], model: str = "test-model") -> None:
        self._responses = list(response_texts)
        self._model = model
        self.calls: list[dict[str, object]] = []
        self.requests: list[LLMRequest] = []

    def default_model(self) -> str:
        return self._model

    def close(self) -> None:
        return None

    def __enter__(self) -> _CaptureSequenceLLMClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "params": params,
            }
        )
        if not self._responses:
            raise AssertionError("No stubbed responses remaining.")
        return LLMResponse(model=model, text=self._responses.pop(0), provider="capture")

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("No stubbed responses remaining.")
        return LLMResponse(
            model=request.model or self.default_model(),
            text=self._responses.pop(0),
            provider="capture",
        )


class _MathToolRuntime(ToolRuntime):
    def __init__(self, *, include_reserved_tool: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._include_reserved_tool = include_reserved_tool

    def list_tools(self) -> Sequence[ToolSpec]:
        specs = [
            ToolSpec(
                name="sum",
                description="Add numbers.",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "additionalProperties": True},
            )
        ]
        if self._include_reserved_tool:
            specs.append(
                ToolSpec(
                    name="skills.activate",
                    description="Reserved name collision.",
                    input_schema={"type": "object", "additionalProperties": True},
                    output_schema={"type": "object", "additionalProperties": True},
                )
            )
        return tuple(specs)

    def invoke(
        self,
        tool_name: str,
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        payload = dict(input)
        self.calls.append((tool_name, payload))
        if tool_name == "sum":
            return ToolResult(
                tool_name="sum",
                ok=True,
                result={"value": int(payload.get("a", 0)) + int(payload.get("b", 0))},
            )
        return ToolResult(tool_name=tool_name, ok=False, error="unknown tool")

    def close(self) -> None:
        return None


def _write_skill(
    root: Path,
    *,
    directory_name: str,
    name: str,
    description: str,
    body: str,
    compatibility: Sequence[str] = (),
    metadata: Mapping[str, str] | None = None,
    allowed_tools: Sequence[str] = (),
) -> Path:
    skill_dir = root / directory_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter_lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
    ]
    if compatibility:
        frontmatter_lines.append("compatibility:\n" + "\n".join(f"  - {item}" for item in compatibility))
    if metadata:
        frontmatter_lines.append("metadata:\n" + "\n".join(f"  {key}: {value}" for key, value in metadata.items()))
    if allowed_tools:
        frontmatter_lines.append("allowed-tools:\n" + "\n".join(f"  - {item}" for item in allowed_tools))
    frontmatter_lines.append("---")
    skill_dir.joinpath("SKILL.md").write_text(
        "\n".join([*frontmatter_lines, body.strip(), ""]),
        encoding="utf-8",
    )
    return skill_dir


@pytest.mark.parametrize(
    ("directory_name", "contents", "match"),
    [
        (
            "drafting",
            "No frontmatter here",
            "must start with YAML frontmatter",
        ),
        (
            "drafting",
            "---\nname: drafting\n---\nbody",
            "must declare a non-empty string 'description'",
        ),
        (
            "drafting",
            '---\nname: "   "\ndescription: blank\n---\nbody',
            "must declare a non-empty string 'name'",
        ),
        (
            "drafting",
            "---\nname: other\ndescription: mismatch\n---\nbody",
            "must match parent directory",
        ),
    ],
)
def test_parse_skill_file_rejects_invalid_skill_definitions(
    tmp_path: Path,
    directory_name: str,
    contents: str,
    match: str,
) -> None:
    skill_dir = tmp_path / directory_name
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        parse_skill_file(skill_file=skill_file, source_label="project")


def test_discover_skills_parses_valid_skill_and_skips_docs_build(tmp_path: Path) -> None:
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(
        skills_root,
        directory_name="research",
        name="research",
        description="Help gather background context.",
        body="Read the references before answering.",
        compatibility=("json", "code"),
        metadata={"owner": "team"},
        allowed_tools=("sum",),
    )
    _write_skill(
        skills_root / "docs" / "_build",
        directory_name="ignored",
        name="ignored",
        description="Should never be discovered.",
        body="Ignore me.",
    )

    catalog = discover_skills(SkillsConfig(project_root=tmp_path))

    assert catalog.names() == ("research",)
    skill = catalog.get("research")
    assert skill is not None
    assert skill.description == "Help gather background context."
    assert skill.compatibility == ("json", "code")
    assert skill.metadata == {"owner": "team"}
    assert skill.allowed_tools == ("sum",)
    assert skill.body == "Read the references before answering."


def test_discover_skills_warns_on_shadowing_and_validates_extra_paths(tmp_path: Path) -> None:
    first_extra = tmp_path / "extra_one"
    second_extra = tmp_path / "extra_two"
    _write_skill(
        first_extra,
        directory_name="analysis",
        name="analysis",
        description="First extra path wins.",
        body="Use the first extra path.",
    )
    _write_skill(
        second_extra,
        directory_name="analysis",
        name="analysis",
        description="Second extra path loses.",
        body="Use the second extra path.",
    )
    _write_skill(
        tmp_path / ".agents" / "skills",
        directory_name="analysis",
        name="analysis",
        description="Project path wins over extras.",
        body="Use the project path.",
    )

    with pytest.warns(UserWarning, match="shadows skill"):
        catalog = discover_skills(
            SkillsConfig(
                project_root=tmp_path,
                extra_paths=("extra_one", "extra_two"),
            )
        )

    resolved = catalog.get("analysis")
    assert resolved is not None
    assert resolved.description == "Project path wins over extras."

    with pytest.raises(ValueError, match="does not exist"):
        discover_skills(
            SkillsConfig(
                project_root=tmp_path,
                extra_paths=("missing-path",),
            )
        )


def test_skills_tool_runtime_adapter_exposes_activation_payload(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path / ".agents" / "skills",
        directory_name="math",
        name="math",
        description="Compute arithmetic carefully.",
        body="Check the calculation and show your work.",
        compatibility=("json",),
        metadata={"owner": "qa"},
        allowed_tools=("other_tool",),
    )
    references_path = skill_dir / "references"
    scripts_path = skill_dir / "scripts"
    assets_path = skill_dir / "assets"
    references_path.mkdir()
    scripts_path.mkdir()
    assets_path.mkdir()
    references_path.joinpath("guide.txt").write_text("reference", encoding="utf-8")
    scripts_path.joinpath("helper.py").write_text("print('hi')", encoding="utf-8")
    assets_path.joinpath("icon.svg").write_text("<svg />", encoding="utf-8")

    adapter = SkillsToolRuntimeAdapter(
        wrapped_runtime=_MathToolRuntime(),
        skills_context=resolve_skills_context(SkillsConfig(project_root=tmp_path)),
    )

    assert {spec.name for spec in adapter.list_tools()} == {"skills.activate", "sum"}
    activation = adapter.invoke(
        "skills.activate",
        {"skill_name": "math"},
        request_id="req-activate",
        dependencies={},
    )

    assert activation.ok is True
    payload = activation.result_dict()
    assert payload["name"] == "math"
    assert payload["description"] == "Compute arithmetic carefully."
    assert payload["compatibility"] == ["json"]
    assert payload["metadata"] == {"owner": "qa"}
    assert payload["allowed_tools"] == ["other_tool"]
    assert payload["instructions"] == "Check the calculation and show your work."
    assert payload["skill_root"] == str(skill_dir.resolve())
    assert payload["resources"]["references"] == [str((references_path / "guide.txt").resolve())]
    assert payload["resources"]["scripts"] == [str((scripts_path / "helper.py").resolve())]
    assert payload["resources"]["assets"] == [str((assets_path / "icon.svg").resolve())]


def test_skills_tool_runtime_adapter_rejects_reserved_name_collision(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / ".agents" / "skills",
        directory_name="math",
        name="math",
        description="Arithmetic helper.",
        body="Use arithmetic.",
    )

    with pytest.raises(ValueError, match=r"reserved tool name 'skills\.activate'"):
        SkillsToolRuntimeAdapter(
            wrapped_runtime=_MathToolRuntime(include_reserved_tool=True),
            skills_context=resolve_skills_context(SkillsConfig(project_root=tmp_path)),
        )


def test_direct_llm_call_injects_pinned_skills_without_catalog(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / ".agents" / "skills",
        directory_name="tone",
        name="tone",
        description="Keep the response concise.",
        body="Prefer short summaries with direct wording.",
    )
    llm_client = _CaptureSequenceLLMClient(response_texts=["done"])
    agent = DirectLLMCall(
        llm_client=llm_client,
        system_prompt="You are concise.",
        skills=SkillsConfig(project_root=tmp_path, pinned_skills=("tone",)),
    )

    result = agent.run("Summarize the report.")

    assert result.success is True
    request = llm_client.requests[0]
    assert '<active_skill name="tone"' in request.messages[0].content
    assert "Available skills:" not in request.messages[0].content
    assert "skills.activate" not in request.messages[1].content
    assert result.metadata["skills"] == {
        "discovered_skill_names": ["tone"],
        "pinned_skill_names": ["tone"],
        "activated_skill_names": ["tone"],
    }


def test_multi_step_json_supports_skill_activation_without_enforcing_allowed_tools(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / ".agents" / "skills",
        directory_name="math",
        name="math",
        description="Use arithmetic helper instructions.",
        body="Call tools to verify arithmetic.",
        allowed_tools=("other_tool",),
    )
    llm_client = _CaptureSequenceLLMClient(
        response_texts=[
            json.dumps(
                {
                    "tool_name": "skills.activate",
                    "tool_input": {"skill_name": "math"},
                    "reason": "load instructions",
                }
            ),
            json.dumps(
                {
                    "tool_name": "sum",
                    "tool_input": {"a": 2, "b": 3},
                    "reason": "apply the skill",
                }
            ),
            json.dumps(
                {
                    "tool_name": "final_answer",
                    "tool_input": {"value": 5},
                    "reason": "done",
                }
            ),
        ]
    )
    tool_runtime = _MathToolRuntime()
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
        skills=SkillsConfig(project_root=tmp_path),
    )

    result = agent.run("Compute 2 + 3.")

    assert result.success is True
    assert result.output["final_output"] == {"value": 5}
    assert [tool_result.tool_name for tool_result in result.tool_results] == ["skills.activate", "sum"]
    assert result.metadata["skills"] == {
        "discovered_skill_names": ["math"],
        "pinned_skill_names": [],
        "activated_skill_names": ["math"],
    }
    first_request = llm_client.requests[0]
    first_prompt = "\n".join(message.content for message in first_request.messages)
    assert "Available skills:" in first_prompt
    assert "skills.activate" in first_prompt
    assert tool_runtime.calls == [("sum", {"a": 2, "b": 3})]


def test_two_speaker_conversation_pattern_injects_pinned_skills(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / ".agents" / "skills",
        directory_name="facilitator",
        name="facilitator",
        description="Keep the discussion moving.",
        body="Ask focused follow-up questions and stay concise.",
    )
    llm_client = _CaptureSequenceLLMClient(response_texts=["a1", "b1"])
    pattern = TwoSpeakerConversationPattern(
        llm_client_a=llm_client,
        max_turns=1,
        skills=SkillsConfig(project_root=tmp_path, pinned_skills=("facilitator",)),
    )

    result = pattern.run("Discuss the rollout plan.")

    assert result.success is True
    assert len(llm_client.requests) == 2
    first_request = llm_client.requests[0]
    assert '<active_skill name="facilitator"' in first_request.messages[0].content
    assert "Available skills:" not in first_request.messages[0].content
    assert result.metadata["skills"] == {
        "discovered_skill_names": ["facilitator"],
        "pinned_skill_names": ["facilitator"],
        "activated_skill_names": ["facilitator"],
    }


def test_router_delegate_pattern_can_activate_skill_then_route(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / ".agents" / "skills",
        directory_name="route_hints",
        name="route_hints",
        description="Hints for choosing the right delegate.",
        body="Prefer alt_two for requests that need escalation.",
    )
    llm_client = _CaptureSequenceLLMClient(
        response_texts=[
            json.dumps(
                {
                    "tool_name": "skills.activate",
                    "tool_input": {"skill_name": "route_hints"},
                    "reason": "load routing guidance",
                }
            ),
            json.dumps(
                {
                    "tool_name": "alt_two",
                    "tool_input": {},
                    "reason": "best fit after reading the skill",
                }
            ),
        ]
    )
    pattern = RouterDelegatePattern(
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        alternatives={
            "alt_one": StaticMarkerAgent(marker="one"),
            "alt_two": StaticMarkerAgent(marker="two"),
        },
        skills=SkillsConfig(project_root=tmp_path),
    )

    result = pattern.run("Route this escalation.")

    assert result.success is True
    assert result.output["details"]["selected_alternative"] == "alt_two"
    assert [tool_result.tool_name for tool_result in result.tool_results] == ["skills.activate", "alt_two"]
    assert result.metadata["skills"] == {
        "discovered_skill_names": ["route_hints"],
        "pinned_skill_names": [],
        "activated_skill_names": ["route_hints"],
    }
    assert len(llm_client.requests) == 2
    first_prompt = "\n".join(message.content for message in llm_client.requests[0].messages)
    second_prompt = "\n".join(message.content for message in llm_client.requests[1].messages)
    assert "Available skills:" in first_prompt
    assert "Activated routing skill:" in second_prompt
