"""Seeded random baseline agent for packaged-problem studies."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, Protocol, cast

from design_research_agents._contracts._delegate import Delegate
from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._termination import (
    TERMINATED_COMPLETED,
    TERMINATED_MAX_STEPS_REACHED,
    TERMINATED_STEP_FAILURE,
)
from design_research_agents._contracts._workflow import LogicStep, WorkflowStepResult
from design_research_agents._implementations._shared._agent_internal._execution_context import (
    resolve_agent_execution_context,
)
from design_research_agents._implementations._shared._agent_internal._result_builders import (
    build_failure_result,
)
from design_research_agents.workflow import CompiledExecution, Workflow

_BASELINE_STEP_ID = "seeded_random_baseline"
_IMPLICIT_DEFAULT_SEED = 0


class _DecisionProblemLike(Protocol):
    """Protocol for decision-style packaged problems."""

    def iter_candidates(self) -> Iterable[object]:
        """Return admissible decision candidates."""


class _GrammarProblemLike(Protocol):
    """Protocol for grammar-style packaged problems."""

    def initial_state(self) -> object:
        """Return the canonical initial grammar state."""

    def enumerate_transitions(self, state: object) -> Iterable[object]:
        """Return admissible transitions from one grammar state."""


class _OptimizationProblemLike(Protocol):
    """Protocol for optimization-style packaged problems."""

    def generate_initial_solution(self, seed: int | None = None) -> object:
        """Return a deterministic or seeded initial candidate."""


class SeededRandomBaselineAgent(Delegate):
    """Seeded control-condition agent for packaged-problem benchmarking.

    The agent uses the same workflow-backed delegate shape as the other public
    agents in this repository. Packaged-problem inputs are supplied through the
    ``dependencies`` mapping at run time.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        grammar_max_steps: int = 1,
    ) -> None:
        """Initialize the baseline agent.

        Args:
            seed: Optional default seed used when per-run seed information is
                not provided through ``dependencies``.
            grammar_max_steps: Maximum number of grammar transitions to sample
                in one grammar rollout.

        Raises:
            ValueError: If ``grammar_max_steps`` is less than one.
        """
        if grammar_max_steps < 1:
            raise ValueError("grammar_max_steps must be >= 1.")
        self._seed = None if seed is None else int(seed)
        self._grammar_max_steps = int(grammar_max_steps)
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one seeded random baseline run."""
        return self.compile(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def compile(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile one seeded random baseline run into a bound workflow."""
        execution_context = resolve_agent_execution_context(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        resolved_request_id = execution_context.request_id
        resolved_dependencies = execution_context.dependencies

        def _run_baseline_step(_context: Mapping[str, object]) -> dict[str, object]:
            return self._execute_baseline(dependencies=resolved_dependencies)

        self.workflow = Workflow(
            steps=[
                LogicStep(
                    step_id=_BASELINE_STEP_ID,
                    handler=_run_baseline_step,
                )
            ]
        )

        return CompiledExecution(
            workflow=self.workflow,
            input=execution_context.prompt,
            request_id=resolved_request_id,
            workflow_request_id=f"{resolved_request_id}:seeded_random_baseline",
            dependencies=resolved_dependencies,
            delegate_name="SeededRandomBaselineAgent",
            trace_input=execution_context.normalized_input,
            finalize=lambda workflow_result: _finalize_baseline_result(
                workflow_result=workflow_result,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
            ),
        )

    def _execute_baseline(
        self,
        *,
        dependencies: Mapping[str, object],
    ) -> dict[str, object]:
        """Execute one seeded random baseline run from normalized dependencies."""
        problem_packet = dependencies.get("problem_packet")
        problem = dependencies.get("problem")
        run_spec = dependencies.get("run_spec")
        condition = dependencies.get("condition")
        explicit_seed = _coerce_seed_like(dependencies.get("seed"))

        problem_object = _resolve_problem_object(problem_packet=problem_packet, problem=problem)
        if problem_object is None:
            raise ValueError(
                "SeededRandomBaselineAgent requires `dependencies['problem']` or "
                "`dependencies['problem_packet']` with `payload['problem_object']`."
            )

        resolved_seed = _resolve_seed(
            explicit_seed=explicit_seed,
            run_spec=run_spec,
            default_seed=self._seed,
        )
        metadata = _build_base_metadata(
            problem_packet=problem_packet,
            problem=problem_object,
            seed=resolved_seed,
            seed_source=_seed_source(
                explicit_seed=explicit_seed,
                run_spec=run_spec,
                default_seed=self._seed,
            ),
            condition=condition,
            run_spec=run_spec,
            grammar_max_steps=self._grammar_max_steps,
        )

        family = _detect_problem_family(problem_object)
        if family == "decision":
            return self._run_decision(
                problem=cast(_DecisionProblemLike, problem_object),
                seed=resolved_seed,
                metadata=metadata,
            )
        if family == "grammar":
            return self._run_grammar(
                problem=cast(_GrammarProblemLike, problem_object),
                seed=resolved_seed,
                metadata=metadata,
            )
        if family == "optimization":
            return self._run_optimization(
                problem=cast(_OptimizationProblemLike, problem_object),
                seed=resolved_seed,
                metadata=metadata,
            )
        raise TypeError(f"Unsupported packaged problem family: {family!r}.")

    def _run_decision(
        self,
        *,
        problem: _DecisionProblemLike,
        seed: int,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        """Sample one decision candidate uniformly at random."""
        candidates = tuple(problem.iter_candidates())
        if not candidates:
            raise ValueError("Decision baseline requires at least one candidate.")

        randomizer = random.Random(seed)
        selected_index = randomizer.randrange(len(candidates))
        selected_candidate = candidates[selected_index]
        final_output = _decision_output_payload(selected_candidate)
        metadata.update(
            {
                "family": "decision",
                "candidate_count": len(candidates),
                "selected_index": selected_index,
                "selected_candidate": _json_safe_value(selected_candidate),
            }
        )
        events = [
            {
                "event_type": "baseline_candidate_selected",
                "actor_id": "agent",
                "text": (f"Selected one seeded random decision candidate from {len(candidates)} admissible options."),
                "meta_json": {
                    "family": "decision",
                    "seed": seed,
                    "selected_index": selected_index,
                },
            }
        ]
        return {
            "final_output": final_output,
            "metrics": {},
            "events": events,
            "run_metadata": metadata,
            "terminated_reason": TERMINATED_COMPLETED,
        }

    def _run_grammar(
        self,
        *,
        problem: _GrammarProblemLike,
        seed: int,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        """Sample a bounded random rollout over public grammar transitions."""
        initial_state = problem.initial_state()
        current_state = initial_state
        randomizer = random.Random(seed)
        transitions_taken: list[dict[str, object]] = []
        events: list[dict[str, object]] = []
        terminated_reason = TERMINATED_MAX_STEPS_REACHED

        for step_number in range(1, self._grammar_max_steps + 1):
            available = tuple(problem.enumerate_transitions(current_state))
            if not available:
                terminated_reason = "no_transitions_available"
                break

            selected_index = randomizer.randrange(len(available))
            selected_transition = available[selected_index]
            next_state = getattr(selected_transition, "next_state", None)
            if next_state is None:
                raise TypeError("Grammar transitions must expose `next_state` for SeededRandomBaselineAgent.")

            trace_entry = {
                "step": step_number,
                "selected_index": selected_index,
                "available_count": len(available),
                "rule_name": str(getattr(selected_transition, "rule_name", "transition")),
                "parameters": _transition_parameters_payload(getattr(selected_transition, "parameters", ())),
            }
            transitions_taken.append(trace_entry)
            events.append(
                {
                    "event_type": "baseline_transition_selected",
                    "actor_id": "agent",
                    "text": f"Applied random grammar transition {trace_entry['rule_name']} at step {step_number}.",
                    "meta_json": trace_entry,
                }
            )
            current_state = next_state

        if not transitions_taken:
            events.append(
                {
                    "event_type": "baseline_transition_skipped",
                    "actor_id": "agent",
                    "text": "No admissible grammar transitions were available from the current state.",
                    "meta_json": {"family": "grammar", "seed": seed},
                }
            )

        steps_executed = len(transitions_taken)
        metadata.update(
            {
                "family": "grammar",
                "initial_state": _json_safe_value(initial_state),
                "transitions": transitions_taken,
                "steps_executed": steps_executed,
                "steps_requested": self._grammar_max_steps,
            }
        )
        return {
            "final_output": {"state": _json_safe_value(current_state)},
            "metrics": {"steps_executed": steps_executed},
            "events": events,
            "run_metadata": metadata,
            "terminated_reason": terminated_reason,
        }

    def _run_optimization(
        self,
        *,
        problem: _OptimizationProblemLike,
        seed: int,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        """Generate one seeded optimization candidate via the public initializer."""
        candidate = _generate_initial_solution(problem=problem, seed=seed)
        metrics: dict[str, object] = {}
        preview_evaluation = None
        preview_error = None
        evaluate = getattr(problem, "evaluate", None)
        if callable(evaluate):
            try:
                preview_evaluation = evaluate(candidate)
            except Exception as exc:  # pragma: no cover - best-effort metadata only.
                preview_error = str(exc)

        if preview_evaluation is not None:
            metrics.update(_optimization_metrics(preview_evaluation))
            metadata["preview_evaluation"] = _json_safe_value(preview_evaluation)
        if preview_error is not None:
            metadata["preview_evaluation_error"] = preview_error

        metadata.update(
            {
                "family": "optimization",
                "candidate": _json_safe_value(candidate),
            }
        )
        events = [
            {
                "event_type": "baseline_candidate_selected",
                "actor_id": "agent",
                "text": "Generated one seeded optimization candidate from the public initializer.",
                "meta_json": {"family": "optimization", "seed": seed},
            }
        ]
        if preview_evaluation is not None:
            events.append(
                {
                    "event_type": "baseline_candidate_evaluated",
                    "actor_id": "system",
                    "text": "Evaluated the sampled optimization candidate for preview metrics.",
                    "meta_json": _optimization_metrics(preview_evaluation),
                }
            )

        return {
            "final_output": {"candidate": _json_safe_value(candidate)},
            "metrics": metrics,
            "events": events,
            "run_metadata": metadata,
            "terminated_reason": TERMINATED_COMPLETED,
        }


def _finalize_baseline_result(
    *,
    workflow_result: ExecutionResult,
    request_id: str,
    dependencies: Mapping[str, object],
) -> ExecutionResult:
    """Finalize one workflow result into the seeded baseline agent contract."""
    workflow_payload = workflow_result.output.get("workflow")
    artifacts = _normalize_sequence(workflow_result.output.get("artifacts"))
    step_result = workflow_result.step_results.get(_BASELINE_STEP_ID)
    if not isinstance(step_result, WorkflowStepResult) or not step_result.success:
        error = "Seeded random baseline workflow failed."
        if isinstance(step_result, WorkflowStepResult) and isinstance(step_result.error, str) and step_result.error:
            error = step_result.error
        return build_failure_result(
            error=error,
            model_response=None,
            tool_results=[],
            request_id=request_id,
            dependencies=dependencies,
            metadata={"stage": _BASELINE_STEP_ID},
            output={
                "final_output": {},
                "metrics": {},
                "events": [],
                "terminated_reason": TERMINATED_STEP_FAILURE,
                "workflow": dict(workflow_payload) if isinstance(workflow_payload, Mapping) else {},
                "artifacts": artifacts,
            },
        )

    run_metadata = step_result.output_dict("run_metadata")
    return ExecutionResult(
        output={
            "final_output": step_result.output_dict("final_output"),
            "metrics": step_result.output_dict("metrics"),
            "events": step_result.output_list("events"),
            "terminated_reason": step_result.terminated_reason or TERMINATED_COMPLETED,
            "workflow": dict(workflow_payload) if isinstance(workflow_payload, Mapping) else {},
            "artifacts": artifacts,
        },
        success=True,
        tool_results=[],
        model_response=None,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            **run_metadata,
        },
    )


def _resolve_problem_object(*, problem_packet: object | None, problem: object | None) -> object | None:
    """Resolve a direct packaged-problem object from supported inputs."""
    if problem is not None:
        if _is_supported_problem_object(problem):
            return problem
        nested_problem = _nested_problem_object(problem)
        if nested_problem is not None:
            return nested_problem

    if problem_packet is None:
        return None
    if _is_supported_problem_object(problem_packet):
        return problem_packet
    return _nested_problem_object(problem_packet)


def _nested_problem_object(container: object) -> object | None:
    """Extract one nested problem object from a packet-like wrapper."""
    direct_problem = _value_from_object(container, "problem_object")
    if _is_supported_problem_object(direct_problem):
        return direct_problem

    payload = _value_from_object(container, "payload")
    if isinstance(payload, Mapping):
        nested_problem = payload.get("problem_object")
        if _is_supported_problem_object(nested_problem):
            return nested_problem
    return None


def _resolve_seed(*, explicit_seed: int | None, run_spec: object | None, default_seed: int | None) -> int:
    """Resolve one deterministic seed using the public precedence contract."""
    if explicit_seed is not None:
        return int(explicit_seed)
    run_spec_seed = _value_from_object(run_spec, "seed")
    normalized_run_spec_seed = _coerce_seed_like(run_spec_seed)
    if normalized_run_spec_seed is not None:
        return normalized_run_spec_seed
    if default_seed is not None:
        return int(default_seed)
    return _IMPLICIT_DEFAULT_SEED


def _seed_source(*, explicit_seed: int | None, run_spec: object | None, default_seed: int | None) -> str:
    """Return the label describing where the resolved seed came from."""
    if explicit_seed is not None:
        return "dependencies.seed"
    if _value_from_object(run_spec, "seed") is not None:
        return "dependencies.run_spec.seed"
    if default_seed is not None:
        return "agent_default"
    return "implicit_default"


def _build_base_metadata(
    *,
    problem_packet: object | None,
    problem: object,
    seed: int,
    seed_source: str,
    condition: object | None,
    run_spec: object | None,
    grammar_max_steps: int,
) -> dict[str, object]:
    """Build the shared metadata block returned by every baseline run."""
    metadata = {
        "agent_name": "SeededRandomBaselineAgent",
        "seed": seed,
        "seed_source": seed_source,
        "grammar_max_steps": grammar_max_steps,
        "problem_class": type(problem).__name__,
        "problem_id": _resolve_problem_id(problem_packet=problem_packet, problem=problem),
    }
    run_id = _value_from_object(run_spec, "run_id")
    if run_id is not None:
        metadata["run_id"] = str(run_id)
    condition_id = _value_from_object(condition, "condition_id")
    if condition_id is not None:
        metadata["condition_id"] = str(condition_id)
    return metadata


def _resolve_problem_id(*, problem_packet: object | None, problem: object) -> str:
    """Return the most specific problem id visible from packet or problem object."""
    packet_problem_id = _value_from_object(problem_packet, "problem_id")
    if packet_problem_id is not None:
        return str(packet_problem_id)

    metadata = _value_from_object(problem, "metadata")
    metadata_problem_id = _value_from_object(metadata, "problem_id")
    if metadata_problem_id is not None:
        return str(metadata_problem_id)

    direct_problem_id = _value_from_object(problem, "problem_id")
    if direct_problem_id is not None:
        return str(direct_problem_id)
    return type(problem).__name__


def _detect_problem_family(problem: object) -> str:
    """Detect the supported packaged-problem family by public duck typing."""
    if callable(getattr(problem, "iter_candidates", None)):
        return "decision"
    if callable(getattr(problem, "initial_state", None)) and callable(getattr(problem, "enumerate_transitions", None)):
        return "grammar"
    if callable(getattr(problem, "generate_initial_solution", None)):
        return "optimization"
    raise TypeError(
        "SeededRandomBaselineAgent supports only decision, grammar, and optimization "
        "problem contracts exposed through public methods."
    )


def _is_supported_problem_object(candidate: object | None) -> bool:
    """Return whether one object exposes a supported public problem contract."""
    if candidate is None:
        return False
    try:
        _detect_problem_family(candidate)
    except TypeError:
        return False
    return True


def _decision_output_payload(candidate: object) -> dict[str, object]:
    """Normalize one decision candidate into a stable mapping payload."""
    if isinstance(candidate, str):
        return {"choice_key": candidate}
    if isinstance(candidate, Mapping):
        return {str(key): _json_safe_value(value) for key, value in candidate.items()}
    values = _value_from_object(candidate, "values")
    if isinstance(values, Mapping):
        return {str(key): _json_safe_value(value) for key, value in values.items()}
    return {"candidate": _json_safe_value(candidate)}


def _transition_parameters_payload(parameters: object) -> dict[str, object]:
    """Normalize one grammar-transition parameter payload."""
    if isinstance(parameters, Mapping):
        return {str(key): _json_safe_value(value) for key, value in parameters.items()}
    if isinstance(parameters, Sequence) and not isinstance(parameters, (str, bytes, bytearray)):
        normalized: dict[str, object] = {}
        for entry in parameters:
            if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes, bytearray)) or len(entry) != 2:
                continue
            key, value = entry
            normalized[str(key)] = _json_safe_value(value)
        return normalized
    return {}


def _generate_initial_solution(*, problem: _OptimizationProblemLike, seed: int) -> object:
    """Call the public optimization initializer with best-effort compatibility."""
    try:
        return problem.generate_initial_solution(seed=seed)
    except TypeError:
        try:
            return cast(Any, problem).generate_initial_solution(seed)
        except TypeError:
            return cast(Any, problem).generate_initial_solution()


def _optimization_metrics(evaluation: object) -> dict[str, object]:
    """Extract stable preview metrics from one optimization evaluation payload."""
    metrics: dict[str, object] = {}
    for field_name in ("objective_value", "total_constraint_violation", "max_constraint_violation"):
        value = _value_from_object(evaluation, field_name)
        if isinstance(value, (int, float)):
            metrics[field_name] = float(value)
    for field_name in ("is_feasible", "higher_is_better"):
        value = _value_from_object(evaluation, field_name)
        if isinstance(value, bool):
            metrics[field_name] = value
    return metrics


def _normalize_sequence(value: object) -> list[object]:
    """Normalize one possibly-sequence payload into a deterministic list."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _json_safe_value(value: object) -> object:
    """Convert one value into a JSON-safe representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return _json_safe_value(cast(Any, value).tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_value(item) for item in value]
    if is_dataclass(value):
        return {field.name: _json_safe_value(getattr(value, field.name)) for field in fields(value)}
    named_tuple_as_dict = getattr(value, "_asdict", None)
    if callable(named_tuple_as_dict):
        return {str(key): _json_safe_value(item) for key, item in named_tuple_as_dict().items()}
    if hasattr(value, "__dict__"):
        public_items = {
            key: item for key, item in vars(value).items() if not key.startswith("_") and not callable(item)
        }
        if public_items:
            return {str(key): _json_safe_value(item) for key, item in public_items.items()}
    return repr(value)


def _value_from_object(source: object | None, name: str) -> object | None:
    """Read one field from a mapping-like or attribute-like object."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _coerce_seed_like(raw_seed: object) -> int | None:
    """Normalize one best-effort seed value into an integer."""
    if isinstance(raw_seed, bool):
        return int(raw_seed)
    if isinstance(raw_seed, int):
        return raw_seed
    if isinstance(raw_seed, float):
        return int(raw_seed)
    if isinstance(raw_seed, str) and raw_seed.strip():
        return int(raw_seed.strip())
    return None


__all__ = ["SeededRandomBaselineAgent"]
