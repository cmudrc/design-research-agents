"""RALPH-style dynamic role loop pattern built on workflow primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._workflow import (
    DelegateBatchCall,
    DelegateBatchStep,
    DelegateTarget,
    LogicStep,
    LoopStep,
)
from design_research_agents._runtime._patterns import (
    MODE_RALPH_LOOP,
    build_compiled_pattern_execution,
    build_loop_callbacks,
    build_pattern_execution_result,
    extract_call_error,
    extract_call_output,
    extract_delegate_batch_call_result,
    is_call_success,
    resolve_pattern_run_context,
    wrap_iteration_handler,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution, Workflow


class RalphLoopPattern(Delegate):
    """Dynamic-role loop with evaluator-threshold termination."""

    @dataclass(slots=True, frozen=True, kw_only=True)
    class RoleSpec:
        """One ordered role participating in each iteration."""

        role_id: str
        delegate: DelegateTarget
        prompt_template: str | None = None

    @dataclass(slots=True, frozen=True, kw_only=True)
    class LoopConfig:
        """Typed loop configuration for Ralph loop execution."""

        max_iterations: int = 3
        consensus_threshold: float = 0.8
        selection_strategy: Literal["best_score", "latest"] = "best_score"

    def __init__(
        self,
        *,
        roles: Sequence[RoleSpec],
        evaluator_role_id: str,
        loop_config: LoopConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store role lineup and loop settings."""
        normalized_roles = self._validate_roles(roles)
        if evaluator_role_id not in {role.role_id for role in normalized_roles}:
            raise ValueError("evaluator_role_id must match one configured role_id.")

        resolved_config = loop_config or self.LoopConfig()
        if resolved_config.max_iterations < 1:
            raise ValueError("loop_config.max_iterations must be >= 1.")
        if not (0.0 <= resolved_config.consensus_threshold <= 1.0):
            raise ValueError("loop_config.consensus_threshold must be in [0, 1].")
        if resolved_config.selection_strategy not in {"best_score", "latest"}:
            raise ValueError("loop_config.selection_strategy must be one of: best_score, latest.")

        self._roles = normalized_roles
        self._evaluator_role_id = evaluator_role_id
        self._loop_config = resolved_config
        self._tracer = tracer
        self.workflow: Workflow | None = None

    @staticmethod
    def _validate_roles(roles: Sequence[RoleSpec]) -> tuple[RoleSpec, ...]:
        """Normalize and validate role list."""
        normalized: list[RalphLoopPattern.RoleSpec] = []
        seen: set[str] = set()
        for role in roles:
            role_id = role.role_id.strip()
            if not role_id:
                raise ValueError("roles must include non-empty role_id values.")
            if role_id in seen:
                raise ValueError("roles must have unique role_id values.")
            seen.add(role_id)
            normalized.append(
                RalphLoopPattern.RoleSpec(
                    role_id=role_id,
                    delegate=role.delegate,
                    prompt_template=role.prompt_template,
                )
            )
        if not normalized:
            raise ValueError("roles must include at least one role.")
        return tuple(normalized)

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute the Ralph loop pattern."""
        return self.compile(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def compile(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile one Ralph loop workflow."""
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=None,
            default_dependencies={},
            request_id=request_id,
            dependencies=dependencies,
        )
        workflow = self._build_workflow(prompt)
        return build_compiled_pattern_execution(
            workflow=workflow,
            pattern_name="RalphLoopPattern",
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            input_payload={
                "prompt": prompt,
                "mode": MODE_RALPH_LOOP,
                "evaluator_role_id": self._evaluator_role_id,
                "max_iterations": self._loop_config.max_iterations,
                "consensus_threshold": self._loop_config.consensus_threshold,
                "selection_strategy": self._loop_config.selection_strategy,
                "roles": [role.role_id for role in self._roles],
            },
            workflow_request_id=f"{run_context.request_id}:ralph_loop_workflow",
            finalize=lambda workflow_result: self._finalize_result(
                workflow_result=workflow_result,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def _build_workflow(self, prompt: str) -> Workflow:
        """Build workflow graph for one Ralph loop run."""

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
            role_results = self._extract_role_results(context)
            failure_role, failure_error = self._find_role_failure(role_results)
            if failure_role is not None:
                loop_state = context.get("loop_state")
                current_state = dict(loop_state) if isinstance(loop_state, Mapping) else {}
                history = (
                    list(current_state.get("iteration_history", []))
                    if isinstance(
                        current_state.get("iteration_history"),
                        list,
                    )
                    else []
                )
                history.append(
                    {
                        "iteration": self._resolve_iteration(context),
                        "roles": _json_ready(role_results),
                        "consensus_score": 0.0,
                        "failed_role": failure_role,
                        "error": failure_error,
                    }
                )
                return {
                    **current_state,
                    "iteration_history": history,
                    "role_outputs": _json_ready(role_results),
                    "failure_role": failure_role,
                    "failure_error": failure_error,
                    "should_continue": False,
                    "terminated_reason": "role_failure",
                }

            consensus_score = self._extract_evaluator_score(role_results)
            synthesized_output = _resolve_synthesized_output(
                role_results,
                ordered_roles=self._roles,
                evaluator_role_id=self._evaluator_role_id,
            )

            loop_state = context.get("loop_state")
            current_state = _as_dict(loop_state)
            history = _as_list(current_state.get("iteration_history"))
            best_score = _safe_float(current_state.get("best_score"))
            best_output = _as_dict(current_state.get("best_output"))
            best_iteration = _safe_int(current_state.get("best_iteration"))

            current_iteration = self._resolve_iteration(context)
            if consensus_score >= best_score:
                best_score = consensus_score
                best_output = dict(synthesized_output)
                best_iteration = current_iteration

            selected_output = (
                dict(synthesized_output) if self._loop_config.selection_strategy == "latest" else dict(best_output)
            )
            should_continue = consensus_score < self._loop_config.consensus_threshold
            terminated_reason = "looping" if should_continue else "consensus_reached"

            history.append(
                {
                    "iteration": current_iteration,
                    "consensus_score": consensus_score,
                    "selected_output": _json_ready(selected_output),
                    "roles": _json_ready(role_results),
                }
            )

            return {
                "role_outputs": _json_ready(role_results),
                "consensus_score": consensus_score,
                "best_score": best_score,
                "best_output": _json_ready(best_output),
                "best_iteration": best_iteration,
                "selected_output": _json_ready(selected_output),
                "iteration_history": history,
                "failure_role": None,
                "failure_error": None,
                "should_continue": should_continue,
                "terminated_reason": terminated_reason,
            }

        wrapped_handler = wrap_iteration_handler(
            _run_iteration,
            error_prefix="RalphLoopPattern iteration",
        )
        loop_callbacks = build_loop_callbacks(
            iteration_step_id="ralph_iteration",
            iteration_handler=wrapped_handler,
        )

        workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
            steps=[
                LoopStep(
                    step_id="ralph_loop",
                    steps=(
                        DelegateBatchStep(
                            step_id="ralph_role_batch",
                            calls_builder=lambda context: self._build_role_calls(
                                prompt=prompt,
                                context=context,
                            ),
                            fail_fast=False,
                        ),
                        LogicStep(
                            step_id="ralph_iteration",
                            dependencies=("ralph_role_batch",),
                            handler=loop_callbacks.iteration_handler,
                        ),
                    ),
                    max_iterations=self._loop_config.max_iterations,
                    initial_state={
                        "role_outputs": {},
                        "consensus_score": 0.0,
                        "best_score": 0.0,
                        "best_output": {},
                        "best_iteration": 0,
                        "selected_output": {},
                        "iteration_history": [],
                        "failure_role": None,
                        "failure_error": None,
                        "should_continue": True,
                        "terminated_reason": "max_iterations_reached",
                    },
                    continue_predicate=loop_callbacks.continue_predicate,
                    state_reducer=loop_callbacks.state_reducer,
                    execution_mode="sequential",
                    failure_policy="propagate_failed_state",
                )
            ],
        )
        self.workflow = workflow
        return workflow

    def _resolve_iteration(self, context: Mapping[str, object]) -> int:
        """Resolve one-based loop iteration from context metadata."""
        loop_meta = context.get("_loop")
        if isinstance(loop_meta, Mapping):
            return max(1, _safe_int(loop_meta.get("iteration", 1)))
        return 1

    def _build_role_calls(
        self,
        *,
        prompt: str,
        context: Mapping[str, object],
    ) -> list[DelegateBatchCall]:
        """Build one delegate call per configured role for current iteration."""
        iteration = self._resolve_iteration(context)
        loop_state = context.get("loop_state")
        state = _as_dict(loop_state)
        selected_output = _as_dict(state.get("selected_output"))
        prior_outputs = _as_dict(state.get("role_outputs"))

        calls: list[DelegateBatchCall] = []
        for role in self._roles:
            prompt_payload = {
                "task": prompt,
                "iteration": iteration,
                "role_id": role.role_id,
                "selected_output": _json_ready(selected_output),
                "prior_role_outputs": _json_ready(prior_outputs),
            }
            default_prompt = json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True)
            if isinstance(role.prompt_template, str) and role.prompt_template.strip():
                try:
                    role_prompt = role.prompt_template.format(
                        task=prompt,
                        iteration=iteration,
                        role_id=role.role_id,
                        selected_output_json=json.dumps(
                            _json_ready(selected_output),
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        prior_role_outputs_json=json.dumps(
                            _json_ready(prior_outputs),
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                    )
                except KeyError as exc:
                    missing = str(exc.args[0]) if exc.args else "unknown"
                    raise ValueError(
                        f"Role prompt template for '{role.role_id}' references unknown variable '{missing}'."
                    ) from exc
            else:
                role_prompt = default_prompt
            calls.append(
                DelegateBatchCall(
                    call_id=role.role_id,
                    delegate=role.delegate,
                    prompt=role_prompt,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                )
            )
        return calls

    def _extract_role_results(self, context: Mapping[str, object]) -> dict[str, dict[str, object]]:
        """Extract role outputs from delegate-batch step result."""
        dependency_results = context.get("dependency_results")
        if not isinstance(dependency_results, Mapping):
            return {}
        batch_payload = dependency_results.get("ralph_role_batch")
        if not isinstance(batch_payload, Mapping):
            return {}
        batch_error = batch_payload.get("error")
        error_text = str(batch_error) if isinstance(batch_error, str) and batch_error.strip() else None
        batch_output = batch_payload.get("output")
        if not isinstance(batch_output, Mapping):
            return self._build_uniform_role_failures(error_text or "Role batch output is missing.")
        batch_results = batch_output.get("results")
        if not isinstance(batch_results, list):
            return self._build_uniform_role_failures(error_text or "Role batch results are missing.")

        normalized_results: dict[str, dict[str, object]] = {}
        raw_result_entries = [entry for entry in batch_results if isinstance(entry, Mapping)]
        for role in self._roles:
            call_result = extract_delegate_batch_call_result(
                results=raw_result_entries,
                call_id=role.role_id,
            )
            if call_result is None:
                normalized_results[role.role_id] = {
                    "success": False,
                    "error": error_text or "Role call result is missing.",
                    "output": {},
                }
                continue
            success = is_call_success(call_result)
            normalized_results[role.role_id] = {
                "success": success,
                "error": (None if success else extract_call_error(call_result, fallback_message="Role call failed.")),
                "output": extract_call_output(call_result),
            }
        return normalized_results

    def _build_uniform_role_failures(self, error_message: str) -> dict[str, dict[str, object]]:
        """Build one failure result for every configured role."""
        return {
            role.role_id: {
                "success": False,
                "error": error_message,
                "output": {},
            }
            for role in self._roles
        }

    def _find_role_failure(
        self,
        role_results: Mapping[str, Mapping[str, object]],
    ) -> tuple[str | None, str | None]:
        """Return first failing role id and message, if any."""
        for role in self._roles:
            result = role_results.get(role.role_id)
            if not isinstance(result, Mapping):
                return role.role_id, "Role result missing."
            if bool(result.get("success", False)):
                continue
            error = result.get("error")
            return role.role_id, str(error) if error is not None else "Role call failed."
        return None, None

    def _extract_evaluator_score(self, role_results: Mapping[str, Mapping[str, object]]) -> float:
        """Extract normalized score from configured evaluator role output."""
        evaluator_result = role_results.get(self._evaluator_role_id)
        if not isinstance(evaluator_result, Mapping):
            return 0.0
        output = evaluator_result.get("output")
        if not isinstance(output, Mapping):
            return 0.0
        return _extract_score(output)

    def _finalize_result(
        self,
        *,
        workflow_result: ExecutionResult,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Build final execution result for one Ralph loop run."""
        loop_step_result = workflow_result.step_results.get("ralph_loop")
        if loop_step_result is None:
            raise RuntimeError("Ralph loop step result is missing.")

        loop_output = loop_step_result.output
        final_state_raw = loop_output.get("final_state")
        final_state = dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}

        failure_role = final_state.get("failure_role")
        failure_error = final_state.get("failure_error")
        terminated_reason = str(
            final_state.get(
                "terminated_reason",
                loop_output.get("terminated_reason", "max_iterations_reached"),
            )
        )
        if terminated_reason == "looping":
            terminated_reason = "max_iterations_reached"

        selected_output = _as_dict(final_state.get("selected_output"))
        best_output = _as_dict(final_state.get("best_output"))
        best_score = _safe_float(final_state.get("best_score"))
        consensus_score = _safe_float(final_state.get("consensus_score"))
        history = _as_list(final_state.get("iteration_history"))

        success = loop_step_result.success and terminated_reason != "role_failure"
        error = None
        if terminated_reason == "role_failure":
            success = False
            role_text = str(failure_role) if isinstance(failure_role, str) and failure_role else "unknown"
            error_text = str(failure_error) if isinstance(failure_error, str) and failure_error else "Role failure."
            error = f"{role_text}: {error_text}"

        return build_pattern_execution_result(
            success=success,
            final_output=dict(selected_output),
            terminated_reason=terminated_reason,
            details={
                "role_outputs": _json_ready(final_state.get("role_outputs", {})),
                "iteration_history": _json_ready(history),
                "consensus_score": consensus_score,
                "best_score": best_score,
                "best_output": _json_ready(best_output),
                "best_iteration": _safe_int(final_state.get("best_iteration")),
                "selection_strategy": self._loop_config.selection_strategy,
                "consensus_threshold": self._loop_config.consensus_threshold,
                "evaluator_role_id": self._evaluator_role_id,
                "roles": [role.role_id for role in self._roles],
            },
            workflow_payload=workflow_result.to_dict(),
            artifacts=workflow_result.output.get("artifacts", []),
            request_id=request_id,
            dependencies=dependencies,
            mode=MODE_RALPH_LOOP,
            metadata={
                "max_iterations": self._loop_config.max_iterations,
                "consensus_threshold": self._loop_config.consensus_threshold,
                "selection_strategy": self._loop_config.selection_strategy,
                "evaluator_role_id": self._evaluator_role_id,
            },
            error=error,
        )


def _resolve_synthesized_output(
    role_results: Mapping[str, Mapping[str, object]],
    *,
    ordered_roles: Sequence[RalphLoopPattern.RoleSpec],
    evaluator_role_id: str,
) -> dict[str, object]:
    """Resolve synthesized output from ordered role outputs."""
    for role in reversed(ordered_roles):
        if role.role_id == evaluator_role_id:
            continue
        result = role_results.get(role.role_id)
        if not isinstance(result, Mapping):
            continue
        output = result.get("output")
        if isinstance(output, Mapping) and output:
            return dict(output)
    for role in reversed(ordered_roles):
        result = role_results.get(role.role_id)
        if not isinstance(result, Mapping):
            continue
        output = result.get("output")
        if isinstance(output, Mapping) and output:
            return dict(output)
    return {}


def _extract_score(output: Mapping[str, object]) -> float:
    """Extract normalized score from role output mapping."""
    score = output.get("score")
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score)))

    model_text = output.get("model_text")
    if isinstance(model_text, str):
        try:
            parsed = json.loads(model_text)
        except json.JSONDecodeError:
            return 0.0
        if isinstance(parsed, Mapping):
            parsed_score = parsed.get("score")
            if isinstance(parsed_score, (int, float)):
                return max(0.0, min(1.0, float(parsed_score)))
    return 0.0


def _safe_float(value: object) -> float:
    """Return float value with deterministic fallback."""
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _safe_int(value: object) -> int:
    """Return int value with deterministic fallback."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _as_dict(value: object) -> dict[str, object]:
    """Return mapping-like values as mutable dictionaries."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_list(value: object) -> list[object]:
    """Return list-like values as mutable lists."""
    if isinstance(value, list):
        return list(value)
    return []


def _json_ready(value: object) -> object:
    """Recursively normalize values into JSON-safe shapes."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["RalphLoopPattern"]
