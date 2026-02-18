from __future__ import annotations

from types import SimpleNamespace

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    LLMMessage,
    LLMRequest,
)
from design_research_agents.contracts.tools import ToolSpec


class DumpObj:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)

    def model_dump(self) -> dict[str, object]:
        return dict(self.__dict__)


class CompletionsStub:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def caps(
    *, streaming: bool = True, tool_calling: str = "native", json_mode: str = "native"
) -> BackendCapabilities:
    return BackendCapabilities(
        streaming=streaming,
        tool_calling=tool_calling,  # type: ignore[arg-type]
        json_mode=json_mode,  # type: ignore[arg-type]
        vision=False,
        max_context_tokens=None,
    )


def request(**overrides: object) -> LLMRequest:
    payload = {
        "messages": [LLMMessage(role="user", content="hello")],
        "model": "gpt-test",
        "temperature": None,
        "max_tokens": None,
        "tools": (),
        "response_schema": None,
        "response_format": None,
        "metadata": {},
        "provider_options": {},
        "task_profile": None,
    }
    payload.update(overrides)
    return LLMRequest(**payload)


def tool(name: str = "calculator") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Compute arithmetic.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


def client_with_completions(completions: object) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))
