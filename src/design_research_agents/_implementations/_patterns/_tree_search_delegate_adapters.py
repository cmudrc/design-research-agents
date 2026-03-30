"""Delegate adapters used by tree-search pattern callable hooks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import TypeGuard

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._workflow import DelegateTarget

GeneratorValue = Mapping[str, object] | str | int | float
GeneratorDelegate = Callable[[Mapping[str, object]], Sequence[GeneratorValue]]
EvaluatorDelegate = Callable[[Mapping[str, object]], float | int | Mapping[str, object]]


def is_workflow_delegate(delegate: object) -> TypeGuard[DelegateTarget]:
    """Return whether ``delegate`` appears to implement workflow-delegate contract."""
    run_callable = getattr(delegate, "run", None)
    return callable(run_callable)


class GeneratorCallableDelegateAdapter:
    """Adapter that wraps callable generator delegates as agent-like delegates."""

    def __init__(self, delegate: GeneratorDelegate) -> None:
        self._delegate = delegate

    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: str = "dag",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del execution_mode, failure_policy, request_id, dependencies
        prompt = ""
        if isinstance(context, Mapping):
            raw_prompt = context.get("prompt")
            prompt = raw_prompt if isinstance(raw_prompt, str) else ""
        parsed_context = _parse_json_context(prompt)
        try:
            raw_children = self._delegate(parsed_context)
        except Exception as exc:
            return ExecutionResult(
                output={"error": str(exc)},
                success=False,
                tool_results=[],
                model_response=None,
                metadata={"delegate_type": "callable_generator"},
            )
        return ExecutionResult(
            output={"candidates": list(raw_children)},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate_type": "callable_generator"},
        )


class EvaluatorCallableDelegateAdapter:
    """Adapter that wraps callable evaluator delegates as agent-like delegates."""

    def __init__(self, delegate: EvaluatorDelegate) -> None:
        self._delegate = delegate

    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: str = "dag",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del execution_mode, failure_policy, request_id, dependencies
        prompt = ""
        if isinstance(context, Mapping):
            raw_prompt = context.get("prompt")
            prompt = raw_prompt if isinstance(raw_prompt, str) else ""
        parsed_context = _parse_json_context(prompt)
        try:
            raw_score = self._delegate(parsed_context)
        except Exception as exc:
            return ExecutionResult(
                output={"error": str(exc)},
                success=False,
                tool_results=[],
                model_response=None,
                metadata={"delegate_type": "callable_evaluator"},
            )
        if isinstance(raw_score, (int, float)):
            output: dict[str, object] = {"score": float(raw_score)}
        elif isinstance(raw_score, Mapping):
            output = dict(raw_score)
        else:
            output = {"score": 0.0}
        return ExecutionResult(
            output=output,
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate_type": "callable_evaluator"},
        )


def _parse_json_context(prompt: str) -> dict[str, object]:
    """Parse delegate prompt payload into mapping context."""
    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        return {"task": prompt}
    if isinstance(parsed, Mapping):
        return {str(key): value for key, value in parsed.items()}
    return {"task": prompt}
