from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FakeProblemMetadata:
    problem_id: str
    title: str
    kind: str


class FakeProblem:
    def __init__(
        self,
        *,
        rendered_brief: str,
        metadata: FakeProblemMetadata,
        statement_markdown: str | None = None,
        brief: str | None = None,
        prompt: str | None = None,
        candidate_kind: str | None = None,
        family: str | None = None,
    ) -> None:
        self._rendered_brief = rendered_brief
        self.metadata = metadata
        self.statement_markdown = statement_markdown
        self.brief = brief
        self.prompt = prompt
        self.candidate_kind = candidate_kind
        self.family = family

    def render_brief(self) -> str:
        return self._rendered_brief
