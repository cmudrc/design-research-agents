"""# Agents / Direct LLM With Pinned Skills.

## Introduction
This example shows how to preload a trusted project-local skill into a one-shot
direct model call. Pinned skills are useful when you want deterministic,
constructor-scoped behavior without exposing automatic activation.


## Technical Implementation
1. Build a ``SkillsConfig`` that points at the current project root and pins one
   trusted local skill name.
2. Construct ``DirectLLMCall`` through the public top-level API and pass the
   skills config at construction time.
3. Execute one direct request so the pinned skill is injected as system-context
   before the user prompt.
4. Print the normalized summary payload for inspection.


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<example-specific payload>",
     "terminated_reason": "<string-or-null>",
     "error": null
   }


## References
- `Agent Skills specification <https://agentskills.io/specification>`_
- `Prompting Guide for deterministic system context <https://platform.openai.com/docs/guides/prompt-engineering>`_
- `System prompting patterns for reliable instruction following <https://www.anthropic.com/engineering/prompt-engineering>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag


def _ensure_example_skill(project_root: Path) -> None:
    skill_dir = project_root / ".agents" / "skills" / "design_brief"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: design_brief",
                "description: Summarize design requirements with concise language.",
                "---",
                "Focus on repairability, clarity, and actionable constraints.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run one direct call with a pinned project-local skill."""
    request_id = "example-direct-llm-pinned-skills-001"
    project_root = Path("artifacts/examples/direct_llm_with_pinned_skills_project")
    _ensure_example_skill(project_root)
    skills = drag.SkillsConfig(
        project_root=project_root,
        pinned_skills=("design_brief",),
    )

    with drag.OpenAICompatibleHTTPLLMClient(
        base_url="http://127.0.0.1:8001/v1",
        default_model="qwen2-1.5b-q4",
    ) as llm_client:
        agent = drag.DirectLLMCall(
            llm_client=llm_client,
            system_prompt="You are a careful design research assistant.",
            skills=skills,
        )
        result = agent.run(
            prompt="Summarize the repairability requirements for a wearable device enclosure.",
            request_id=request_id,
        )

    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
