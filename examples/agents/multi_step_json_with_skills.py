"""# Agents / Multi Step JSON With Skills.

## Introduction
This example shows a tool-capable multi-step agent with Agent Skills enabled.
Discovered project-local skills can be activated on demand with
``skills.activate`` before the agent selects one of the regular tools.


## Technical Implementation
1. Build a ``SkillsConfig`` rooted at the current project so the agent can
   discover local ``.agents/skills`` definitions.
2. Construct ``MultiStepAgent`` in JSON mode with the public ``Toolbox`` facade.
3. Allow the model to activate a discovered skill before making a regular tool
   call and explicit final answer.
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
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag


def _ensure_example_skill(project_root: Path) -> None:
    skill_dir = project_root / ".agents" / "skills" / "word_count_helper"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: word_count_helper",
                "description: Use text.word_count before finalizing numeric answers.",
                "allowed-tools:",
                "  - text.word_count",
                "---",
                "Activate this skill before calling text.word_count on short phrases.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run one JSON tool-calling agent with skills discovery enabled."""
    request_id = "example-multi-step-json-skills-001"
    project_root = Path("artifacts/examples/multi_step_json_with_skills_project")
    _ensure_example_skill(project_root)
    skills = drag.SkillsConfig(project_root=project_root)

    with drag.Toolbox() as tool_runtime, drag.LlamaCppServerLLMClient() as llm_client:
        agent = drag.MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=4,
            allowed_tools=("text.word_count",),
            skills=skills,
        )
        result = agent.run(
            prompt=(
                "Activate a relevant skill if one is available, then use text.word_count to count "
                "the words in 'design research agents'. Return only the word_count."
            ),
            request_id=request_id,
        )

    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
