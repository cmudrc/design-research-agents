"""Nominal-team pattern with independent generation and best-of-N selection."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from string import Formatter
from typing import cast

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._workflow import (
    DelegateBatchCall,
    DelegateBatchStep,
    DelegateTarget,
    LogicStep,
)
from design_research_agents._runtime._common._delegate_invocation import invoke_delegate
from design_research_agents._runtime._patterns import (
    MODE_NOMINAL_TEAM,
    build_compiled_pattern_execution,
    build_pattern_execution_result,
    extract_call_error,
    extract_call_output,
    extract_delegate_batch_call_result,
    is_call_success,
    resolve_pattern_run_context,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution, Workflow

SelectionValue = Mapping[str, object] | Sequence[float | int] | str | int
SelectionEvaluator = Callable[[Mapping[str, object]], SelectionValue]

_MEMBER_TEMPLATE_FIELDS = frozenset({"task", "member_id"})


class NominalTeamPattern(Delegate):
    """Independent member generation followed by evaluator-based best-of-N selection."""

    @dataclass(slots=True, frozen=True, kw_only=True)
    class MemberSpec:
        """One independent nominal-team member configuration."""

        member_id: str
        delegate: DelegateTarget
        prompt_template: str | None = None

    def __init__(
        self,
        *,
        team_members: Sequence[MemberSpec],
        evaluator_delegate: DelegateTarget | SelectionEvaluator,
        tracer: Tracer | None = None,
    ) -> None:
        """Store team-member and evaluator delegates."""
        self._team_members = self._validate_team_members(team_members)
        self._evaluator_delegate = _normalize_evaluator_delegate(evaluator_delegate)
        self._tracer = tracer
        self.workflow: Workflow | None = None

    @staticmethod
    def _validate_team_members(team_members: Sequence[MemberSpec]) -> tuple[MemberSpec, ...]:
        """Normalize configured team members and validate prompt templates."""
        normalized: list[NominalTeamPattern.MemberSpec] = []
        seen_member_ids: set[str] = set()
        for member in team_members:
            member_id = member.member_id.strip()
            if not member_id:
                raise ValueError("team_members must include non-empty member_id values.")
            if member_id in seen_member_ids:
                raise ValueError("team_members must have unique member_id values.")
            seen_member_ids.add(member_id)
            _validate_member_prompt_template(prompt_template=member.prompt_template, member_id=member_id)
            normalized.append(
                NominalTeamPattern.MemberSpec(
                    member_id=member_id,
                    delegate=member.delegate,
                    prompt_template=member.prompt_template,
                )
            )
        if not normalized:
            raise ValueError("team_members must include at least one member.")
        return tuple(normalized)

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one nominal-team run."""
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
        """Compile one nominal-team workflow."""
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=None,
            default_dependencies={},
            request_id=request_id,
            dependencies=dependencies,
        )
        workflow = self._build_workflow(
            prompt,
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
        )
        return build_compiled_pattern_execution(
            workflow=workflow,
            pattern_name="NominalTeamPattern",
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            input_payload={
                "prompt": prompt,
                "mode": MODE_NOMINAL_TEAM,
                "team_members": [member.member_id for member in self._team_members],
            },
            workflow_request_id=f"{run_context.request_id}:nominal_team_workflow",
            failure_policy="propagate_failed_state",
            finalize=lambda workflow_result: _build_nominal_team_result(
                workflow_result=workflow_result,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
                team_members=self._team_members,
            ),
        )

    def _build_workflow(
        self,
        prompt: str,
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> Workflow:
        """Build the nominal-team generation and evaluation workflow."""
        workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
            default_failure_policy="propagate_failed_state",
            steps=[
                DelegateBatchStep(
                    step_id="nominal_team_generation",
                    calls_builder=lambda _context: self._build_generation_calls(prompt=prompt),
                    fail_fast=False,
                ),
                LogicStep(
                    step_id="nominal_team_selection",
                    dependencies=("nominal_team_generation",),
                    handler=lambda context: self._run_selection(
                        prompt=prompt,
                        context=context,
                        request_id=request_id,
                        dependencies=dependencies,
                    ),
                ),
            ],
        )
        self.workflow = workflow
        return workflow

    def _build_generation_calls(self, *, prompt: str) -> list[DelegateBatchCall]:
        """Build one independent delegate call for each nominal-team member."""
        calls: list[DelegateBatchCall] = []
        for member in self._team_members:
            prompt_payload = {
                "task": prompt,
                "member_id": member.member_id,
                "mode": MODE_NOMINAL_TEAM,
            }
            default_prompt = json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True)
            if isinstance(member.prompt_template, str) and member.prompt_template.strip():
                member_prompt = member.prompt_template.format(
                    task=prompt,
                    member_id=member.member_id,
                )
            else:
                member_prompt = default_prompt
            calls.append(
                DelegateBatchCall(
                    call_id=member.member_id,
                    delegate=member.delegate,
                    prompt=member_prompt,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                )
            )
        return calls

    def _run_selection(
        self,
        *,
        prompt: str,
        context: Mapping[str, object],
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Normalize generation outputs, call evaluator, and select one candidate."""
        generation_result = _extract_generation_result(context)
        if generation_result is None:
            return _build_selection_failure(
                team_members=self._team_members,
                terminated_reason="generation_failure",
                error="Nominal-team generation result is missing.",
            )

        member_results = _extract_member_results(
            generation_result=generation_result,
            ordered_members=self._team_members,
        )
        eligible_candidates = [result for result in member_results if bool(result.get("candidate_ready", False))]
        if not eligible_candidates:
            return _build_selection_failure(
                team_members=self._team_members,
                candidate_results=member_results,
                terminated_reason="no_candidates",
                error="No nominal-team member produced a selectable candidate.",
            )

        evaluator_prompt = json.dumps(
            {
                "task": prompt,
                "mode": MODE_NOMINAL_TEAM,
                "candidate_count": len(eligible_candidates),
                "candidates": [
                    {
                        "candidate_index": _safe_int(candidate.get("candidate_index")),
                        "team_index": _safe_int(candidate.get("team_index")),
                        "member_id": str(candidate.get("member_id", "")),
                        "candidate": _json_ready(candidate.get("candidate", {})),
                        "output": _json_ready(candidate.get("output", {})),
                    }
                    for candidate in eligible_candidates
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
        )

        evaluator_invocation = invoke_delegate(
            delegate=self._evaluator_delegate,
            prompt=evaluator_prompt,
            step_context=context,
            request_id=f"{request_id}:nominal_team:evaluator",
            execution_mode="sequential",
            failure_policy="skip_dependents",
            dependencies=dependencies,
        )
        evaluator_result = evaluator_invocation.result
        evaluator_output = _normalize_evaluator_output(evaluator_result.output)
        if not evaluator_result.success:
            return _build_selection_failure(
                team_members=self._team_members,
                candidate_results=member_results,
                terminated_reason="evaluation_failure",
                evaluator_output=evaluator_output,
                error=_extract_execution_error(
                    result=evaluator_result,
                    fallback="Nominal-team evaluator failed.",
                ),
            )

        selected_result = _resolve_selected_result(
            evaluator_output=evaluator_output,
            candidate_results=eligible_candidates,
        )
        if selected_result is None:
            return _build_selection_failure(
                team_members=self._team_members,
                candidate_results=member_results,
                terminated_reason="evaluation_failure",
                evaluator_output=evaluator_output,
                error=(
                    "Evaluator output must identify one candidate via best_index, selected_index, "
                    "best_member_id, selected_member_id, or numeric scores."
                ),
            )

        selected_member_id = str(selected_result.get("member_id", ""))
        selected_candidate_index = _safe_int(selected_result.get("candidate_index"))
        selected_team_index = _safe_int(selected_result.get("team_index"))
        scores = _normalize_score_map(
            evaluator_output=evaluator_output,
            candidate_results=eligible_candidates,
        )
        selection_score = scores.get(selected_member_id)

        return {
            "success": True,
            "terminated_reason": "selected",
            "selected_candidate": _json_ready(selected_result.get("candidate", {})),
            "selected_member_id": selected_member_id,
            "selected_candidate_index": selected_candidate_index,
            "selected_team_index": selected_team_index,
            "candidate_count": len(eligible_candidates),
            "candidate_results": _json_ready(member_results),
            "scores": scores,
            "selection_score": selection_score,
            "evaluator_output": _json_ready(evaluator_output),
            "error": None,
        }


class _NominalTeamEvaluatorCallableAdapter:
    """Adapter that wraps evaluator callables as workflow-runner delegates."""

    def __init__(self, evaluator: SelectionEvaluator) -> None:
        self._evaluator = evaluator

    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: str = "dag",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one callable evaluator over parsed JSON selection context."""
        del execution_mode, failure_policy, request_id, dependencies
        prompt = ""
        if isinstance(context, Mapping):
            raw_prompt = context.get("prompt")
            prompt = raw_prompt if isinstance(raw_prompt, str) else ""
        parsed_context = _parse_json_context(prompt)
        try:
            raw_selection = self._evaluator(parsed_context)
        except Exception as exc:
            return ExecutionResult(
                output={"error": str(exc)},
                success=False,
                tool_results=[],
                model_response=None,
                metadata={"delegate_type": "callable_nominal_team_evaluator"},
            )

        if isinstance(raw_selection, Mapping):
            output: dict[str, object] = dict(raw_selection)
        elif isinstance(raw_selection, str):
            output = {"best_member_id": raw_selection}
        elif isinstance(raw_selection, int):
            output = {"best_index": raw_selection}
        elif isinstance(raw_selection, Sequence) and not isinstance(raw_selection, (str, bytes, bytearray)):
            output = {"scores": [value for value in raw_selection if isinstance(value, (int, float))]}
        else:
            output = {}
        return ExecutionResult(
            output=output,
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate_type": "callable_nominal_team_evaluator"},
        )


def _normalize_evaluator_delegate(
    evaluator_delegate: DelegateTarget | SelectionEvaluator,
) -> DelegateTarget:
    """Normalize evaluator delegate into one workflow-compatible target."""
    if _is_delegate_like(evaluator_delegate):
        return cast(DelegateTarget, evaluator_delegate)
    return _NominalTeamEvaluatorCallableAdapter(cast(SelectionEvaluator, evaluator_delegate))


def _validate_member_prompt_template(*, prompt_template: str | None, member_id: str) -> None:
    """Reject unsupported prompt-template variables at construction time."""
    if not isinstance(prompt_template, str) or not prompt_template.strip():
        return
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(prompt_template):
        if not field_name:
            continue
        root_field = field_name.split(".", 1)[0].split("[", 1)[0]
        if root_field in _MEMBER_TEMPLATE_FIELDS:
            continue
        raise ValueError(f"Member prompt template for '{member_id}' references unknown variable '{root_field}'.")


def _build_nominal_team_result(
    *,
    workflow_result: ExecutionResult,
    request_id: str,
    dependencies: Mapping[str, object],
    team_members: Sequence[NominalTeamPattern.MemberSpec],
) -> ExecutionResult:
    """Build the final nominal-team result from workflow execution payloads."""
    selection_step = workflow_result.step_results.get("nominal_team_selection")
    if selection_step is None:
        return build_pattern_execution_result(
            success=False,
            final_output={},
            terminated_reason="generation_failure",
            details={
                "team_members": [member.member_id for member in team_members],
                "candidate_count": 0,
                "candidate_results": [],
                "selected_member_id": None,
                "selected_candidate_index": None,
                "selected_team_index": None,
                "scores": {},
                "selection_score": None,
                "evaluator_output": {},
            },
            workflow_payload=workflow_result.to_dict(),
            artifacts=workflow_result.output.get("artifacts", []),
            request_id=request_id,
            dependencies=dependencies,
            mode=MODE_NOMINAL_TEAM,
            metadata={"team_members": [member.member_id for member in team_members]},
            error="Nominal-team selection step result is missing.",
        )

    selection_output = dict(selection_step.output) if isinstance(selection_step.output, Mapping) else {}
    terminated_reason = str(selection_output.get("terminated_reason", "generation_failure"))
    error = selection_output.get("error")
    error_text = str(error) if isinstance(error, str) and error.strip() else selection_step.error

    return build_pattern_execution_result(
        success=bool(selection_output.get("success", False)) and selection_step.success,
        final_output=_as_dict(selection_output.get("selected_candidate")),
        terminated_reason=terminated_reason,
        details={
            "team_members": [member.member_id for member in team_members],
            "candidate_count": _safe_int(selection_output.get("candidate_count")),
            "candidate_results": _as_list(selection_output.get("candidate_results")),
            "selected_member_id": selection_output.get("selected_member_id"),
            "selected_candidate_index": _value_or_none(selection_output.get("selected_candidate_index")),
            "selected_team_index": _value_or_none(selection_output.get("selected_team_index")),
            "scores": _as_dict(selection_output.get("scores")),
            "selection_score": _value_or_none(selection_output.get("selection_score")),
            "evaluator_output": _as_dict(selection_output.get("evaluator_output")),
        },
        workflow_payload=workflow_result.to_dict(),
        artifacts=workflow_result.output.get("artifacts", []),
        request_id=request_id,
        dependencies=dependencies,
        mode=MODE_NOMINAL_TEAM,
        metadata={
            "team_members": [member.member_id for member in team_members],
            "selected_member_id": selection_output.get("selected_member_id"),
            "candidate_count": _safe_int(selection_output.get("candidate_count")),
        },
        error=error_text,
    )


def _build_selection_failure(
    *,
    team_members: Sequence[NominalTeamPattern.MemberSpec],
    terminated_reason: str,
    error: str,
    candidate_results: Sequence[Mapping[str, object]] | None = None,
    evaluator_output: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one normalized failure payload for the selection logic step."""
    results = list(candidate_results or [])
    candidate_count = sum(1 for result in results if bool(result.get("candidate_ready", False)))
    return {
        "success": False,
        "terminated_reason": terminated_reason,
        "selected_candidate": {},
        "selected_member_id": None,
        "selected_candidate_index": None,
        "selected_team_index": None,
        "candidate_count": candidate_count,
        "candidate_results": _json_ready(results),
        "scores": {},
        "selection_score": None,
        "evaluator_output": _json_ready(evaluator_output or {}),
        "team_members": [member.member_id for member in team_members],
        "error": error,
    }


def _extract_generation_result(context: Mapping[str, object]) -> Mapping[str, object] | None:
    """Return the generation-step payload from logic-step dependency results."""
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return None
    generation_result = dependency_results.get("nominal_team_generation")
    if isinstance(generation_result, Mapping):
        return generation_result
    return None


def _extract_member_results(
    *,
    generation_result: Mapping[str, object],
    ordered_members: Sequence[NominalTeamPattern.MemberSpec],
) -> list[dict[str, object]]:
    """Normalize batch-member results in declared team-member order."""
    batch_output = generation_result.get("output")
    results_payload = batch_output.get("results") if isinstance(batch_output, Mapping) else None
    raw_results = (
        [result for result in results_payload if isinstance(result, Mapping)]
        if isinstance(results_payload, list)
        else []
    )

    normalized: list[dict[str, object]] = []
    candidate_index = 0
    for team_index, member in enumerate(ordered_members):
        call_result = extract_delegate_batch_call_result(results=raw_results, call_id=member.member_id)
        output = extract_call_output(call_result)
        call_success = is_call_success(call_result)
        candidate = _extract_candidate(output)
        candidate_ready = call_success and bool(candidate)

        error = None
        if not call_success:
            error = extract_call_error(call_result, fallback_message="Nominal-team member call failed.")
        elif not candidate_ready:
            error = "Nominal-team member output did not include a selectable candidate."

        normalized_result = {
            "member_id": member.member_id,
            "team_index": team_index,
            "success": call_success,
            "candidate_ready": candidate_ready,
            "candidate_index": candidate_index if candidate_ready else None,
            "error": error,
            "output": _json_ready(output),
            "candidate": _json_ready(candidate if candidate_ready else {}),
        }
        normalized.append(normalized_result)
        if candidate_ready:
            candidate_index += 1

    return normalized


def _extract_candidate(output: Mapping[str, object]) -> dict[str, object]:
    """Extract one candidate payload from a delegate output mapping."""
    final_output = output.get("final_output")
    if final_output is not None:
        candidate = _normalize_candidate_value(final_output)
        if candidate:
            return candidate

    direct_candidate = output.get("candidate")
    if direct_candidate is not None:
        candidate = _normalize_candidate_value(direct_candidate)
        if candidate:
            return candidate

    model_text = output.get("model_text")
    if isinstance(model_text, str):
        candidate = _parse_candidate_from_text(model_text)
        if candidate:
            return candidate

    if _has_meaningful_output(output):
        return {str(key): _json_ready(value) for key, value in output.items()}
    return {}


def _normalize_evaluator_output(output: Mapping[str, object]) -> dict[str, object]:
    """Normalize evaluator output to one flat mapping used for selection parsing."""
    normalized = _normalize_selection_payload(output.get("final_output")) or dict(output)
    if _has_selection_signal(normalized):
        return normalized

    model_text = output.get("model_text")
    if isinstance(model_text, str):
        return _parse_evaluator_text(model_text) or normalized
    return normalized


def _normalize_selection_payload(value: object) -> dict[str, object] | None:
    """Normalize supported evaluator payloads into one selection mapping."""
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, list):
        return {"scores": list(value)}
    if isinstance(value, str):
        return _parse_evaluator_text(value)
    if isinstance(value, int):
        return {"best_index": value}
    return None


def _parse_evaluator_text(text: str) -> dict[str, object]:
    """Parse evaluator JSON text or fall back to a direct member-id selection."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        normalized_text = text.strip()
        return {"best_member_id": normalized_text} if normalized_text else {}
    return _normalize_selection_payload(parsed) or {}


def _resolve_selected_result(
    *,
    evaluator_output: Mapping[str, object],
    candidate_results: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Resolve one selected candidate result from evaluator output."""
    member_id_lookup = {
        str(result.get("member_id", "")): dict(result) for result in candidate_results if isinstance(result, Mapping)
    }
    candidate_index_lookup = {
        _safe_int(result.get("candidate_index")): dict(result)
        for result in candidate_results
        if isinstance(result, Mapping) and result.get("candidate_index") is not None
    }

    for key in ("best_index", "selected_index", "winner_index", "candidate_index"):
        candidate_index = _coerce_optional_int(evaluator_output.get(key))
        if candidate_index is None:
            continue
        selected = candidate_index_lookup.get(candidate_index)
        if selected is not None:
            return selected

    for key in ("best_member_id", "selected_member_id", "winner", "member_id"):
        raw_member_id = evaluator_output.get(key)
        if not isinstance(raw_member_id, str):
            continue
        selected = member_id_lookup.get(raw_member_id.strip())
        if selected is not None:
            return selected

    scores = _normalize_score_map(
        evaluator_output=evaluator_output,
        candidate_results=candidate_results,
    )
    if not scores:
        return None

    selected_member_id = max(
        scores,
        key=lambda member_id: (
            scores[member_id],
            -_safe_int(member_id_lookup[member_id].get("candidate_index")),
            member_id,
        ),
    )
    selected = member_id_lookup.get(selected_member_id)
    if selected is None:
        return None
    return selected


def _normalize_score_map(
    *,
    evaluator_output: Mapping[str, object],
    candidate_results: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Normalize evaluator scores to a member-id keyed mapping."""
    raw_scores = evaluator_output.get("scores")
    if isinstance(raw_scores, Mapping):
        normalized: dict[str, float] = {}
        for result in candidate_results:
            member_id = str(result.get("member_id", ""))
            score = _coerce_optional_float(raw_scores.get(member_id))
            if member_id and score is not None:
                normalized[member_id] = score
        return normalized

    if isinstance(raw_scores, Sequence) and not isinstance(raw_scores, (str, bytes, bytearray)):
        normalized = {}
        candidate_list = [dict(result) for result in candidate_results if isinstance(result, Mapping)]
        for index, raw_score in enumerate(raw_scores):
            if index >= len(candidate_list):
                break
            score = _coerce_optional_float(raw_score)
            if score is None:
                continue
            member_id = str(candidate_list[index].get("member_id", ""))
            if member_id:
                normalized[member_id] = score
        return normalized

    return {}


def _has_selection_signal(output: Mapping[str, object]) -> bool:
    """Return whether evaluator output already exposes selection keys."""
    return any(
        key in output
        for key in (
            "best_index",
            "selected_index",
            "winner_index",
            "candidate_index",
            "best_member_id",
            "selected_member_id",
            "winner",
            "member_id",
            "scores",
        )
    )


def _parse_json_context(prompt: str) -> dict[str, object]:
    """Parse JSON string payloads into mapping context with string keys."""
    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        return {"task": prompt}
    if isinstance(parsed, Mapping):
        return {str(key): value for key, value in parsed.items()}
    return {"task": prompt}


def _parse_candidate_from_text(model_text: str) -> dict[str, object]:
    """Parse one candidate payload from model text or fall back to plain text."""
    try:
        parsed = json.loads(model_text)
    except json.JSONDecodeError:
        normalized_text = model_text.strip()
        return {"text": normalized_text} if normalized_text else {}
    candidate = _normalize_candidate_value(parsed)
    if candidate:
        return candidate
    normalized_text = model_text.strip()
    return {"text": normalized_text} if normalized_text else {}


def _normalize_candidate_value(value: object) -> dict[str, object]:
    """Normalize heterogeneous candidate values into a mapping."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (str, int, float)):
        return {"value": _json_ready(value)}
    return {}


def _has_meaningful_output(output: Mapping[str, object]) -> bool:
    """Return whether output carries non-empty candidate-like content."""
    for value in output.values():
        if isinstance(value, Mapping) and value:
            return True
        if isinstance(value, (list, tuple)) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (int, float, bool)):
            return True
    return False


def _extract_execution_error(*, result: ExecutionResult, fallback: str) -> str:
    """Extract a human-readable error message from one execution result."""
    error = result.error
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(result.output, Mapping):
        output_error = result.output.get("error")
        if isinstance(output_error, str) and output_error.strip():
            return output_error.strip()
    return fallback


def _is_delegate_like(value: object) -> bool:
    """Return whether value already appears to implement a delegate contract."""
    return callable(getattr(value, "run", None))


def _coerce_optional_int(value: object) -> int | None:
    """Coerce supported values to int when possible."""
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
            return None
    return None


def _coerce_optional_float(value: object) -> float | None:
    """Coerce supported values to float when possible."""
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _safe_int(value: object) -> int:
    """Return int with deterministic fallback to zero."""
    normalized = _coerce_optional_int(value)
    return normalized if normalized is not None else 0


def _as_dict(value: object) -> dict[str, object]:
    """Return mapping-like values as mutable dictionaries."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_list(value: object) -> list[object]:
    """Return list-like values as mutable lists."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _value_or_none(value: object) -> object | None:
    """Normalize absent sentinel values to ``None``."""
    if value is None:
        return None
    return value


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


__all__ = ["NominalTeamPattern"]
