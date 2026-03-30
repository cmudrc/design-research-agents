"""Shared loop-callback builders for workflow-native pattern implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._workflow import (
    LoopStepContinuePredicate,
    LoopStepStateReducer,
)

LoopIterationHandler = Callable[[Mapping[str, object]], Mapping[str, object]]
"""Callable that computes one loop-iteration output mapping from step context."""


@dataclass(slots=True, frozen=True)
class LoopCallbacks:
    """Typed callback bundle consumed by ``LoopStep`` configuration."""

    continue_predicate: LoopStepContinuePredicate
    iteration_handler: LoopIterationHandler
    state_reducer: LoopStepStateReducer


def wrap_iteration_handler(
    handler: Callable[[Mapping[str, object]], Mapping[str, object] | object],
    *,
    error_prefix: str,
) -> LoopIterationHandler:
    """Wrap iteration handlers with deterministic mapping normalization."""

    def _wrapped(context: Mapping[str, object]) -> Mapping[str, object]:
        output = handler(context)
        if isinstance(output, Mapping):
            return dict(output)
        raise ValueError(f"{error_prefix} must return a mapping.")

    return _wrapped


def build_loop_callbacks(
    *,
    iteration_step_id: str,
    iteration_handler: LoopIterationHandler,
    continue_predicate: LoopStepContinuePredicate | None = None,
    state_reducer: LoopStepStateReducer | None = None,
    continue_key: str = "should_continue",
) -> LoopCallbacks:
    """Build loop callbacks with shared continuation/reducer defaults."""
    resolved_continue = continue_predicate or _default_continue_predicate(continue_key=continue_key)
    resolved_reducer = state_reducer or _default_state_reducer(iteration_step_id=iteration_step_id)
    return LoopCallbacks(
        continue_predicate=resolved_continue,
        iteration_handler=iteration_handler,
        state_reducer=resolved_reducer,
    )


def _default_continue_predicate(*, continue_key: str) -> LoopStepContinuePredicate:
    """Build default continue predicate from one loop-state key."""

    def _predicate(iteration: int, state: Mapping[str, object]) -> bool:
        del iteration
        return bool(state.get(continue_key, True))

    return _predicate


def _default_state_reducer(*, iteration_step_id: str) -> LoopStepStateReducer:
    """Build default reducer that forwards one iteration-step output mapping."""

    def _reducer(
        state: Mapping[str, object],
        iteration_result: ExecutionResult,
        iteration: int,
    ) -> Mapping[str, object]:
        del iteration
        iteration_step = iteration_result.step_results.get(iteration_step_id)
        if iteration_step is None or not getattr(iteration_step, "success", False):
            return dict(state)
        output = getattr(iteration_step, "output", {})
        return dict(output) if isinstance(output, Mapping) else dict(state)

    return _reducer


__all__ = [
    "LoopCallbacks",
    "LoopIterationHandler",
    "build_loop_callbacks",
    "wrap_iteration_handler",
]
