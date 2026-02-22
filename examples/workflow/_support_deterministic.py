"""Local deterministic delegates for workflow examples."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.workflow import ExecutionResult


def _build_execution_result(
    *,
    output: dict[str, object],
    metadata: Mapping[str, object] | None = None,
) -> ExecutionResult:
    """Build one execution-result instance with deterministic payload."""
    return ExecutionResult(
        success=True,
        output=output,
        tool_results=[],
        model_response=None,
        metadata=dict(metadata or {}),
    )


class FixedDesignPeerAgent:
    """Deterministic peer delegate returning fixed contribution payload."""

    def __init__(self, *, messages: list[str], stop: bool = False) -> None:
        self._messages = list(messages)
        self._stop = stop

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return _build_execution_result(
            output={
                "messages": list(self._messages),
                "proposals": {},
                "decisions": {},
                "stop": self._stop,
            },
            metadata={"delegate": "fixed-design-peer"},
        )


class EchoDesignReasoningAgent:
    """Deterministic local reasoning delegate for RAG examples."""

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del request_id, dependencies
        return _build_execution_result(
            output={
                "summary": "Produced design recommendation from retrieved context.",
                "recommendation": "Prioritize maintainability checks and explicit testability criteria.",
                "prompt_chars": len(prompt),
            },
            metadata={"delegate": "echo-design"},
        )
