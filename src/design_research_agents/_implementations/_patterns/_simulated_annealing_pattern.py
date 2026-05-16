"""Reusable ``simulated_annealing`` orchestration scaffold."""

from __future__ import annotations

import math
import random
import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Literal

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._workflow import LogicStep, LoopStep
from design_research_agents._runtime._patterns import (
    MODE_SIMULATED_ANNEALING,
    build_compiled_pattern_execution,
    build_loop_callbacks,
    build_pattern_execution_result,
    resolve_pattern_run_context,
    wrap_iteration_handler,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution, Workflow

NeighborDelegate = Callable[[Mapping[str, object]], Mapping[str, object]]
ModificationsDelegate = Callable[[Mapping[str, object]], list[Mapping[str, object]]]
ObjectiveDelegate = Callable[[Mapping[str, object]], float]
ConstraintDelegate = Callable[[Mapping[str, object]], bool]
InitialStateGenerator = Callable[[], Mapping[str, object]]
StateValidator = Callable[[Mapping[str, object]], bool]


class TemperatureSchedule(ABC):
    """Base class for temperature decay schedules."""

    @abstractmethod
    def get_temperature(
        self,
        initial_temperature: float,
        iteration: int,
        *,
        current_temperature: float | None = None,
        objective_value_history: list[float] | None = None,
    ) -> float:
        """Return the temperature for one iteration.

        Args:
            initial_temperature: The initial temperature configured for the SA run.
            iteration: The current iteration number (starting from 0).
            current_temperature: The temperature from the previous iteration, if applicable.
            objective_value_history: List of objective values from previous iterations, if applicable.

        Returns:
            Temperature value for current iteration.
        """


class LinearSchedule(TemperatureSchedule):
    """Linear decay schedule."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def get_temperature(
        self,
        initial_temperature: float,
        iteration: int,
        *,
        current_temperature: float | None = None,
        objective_value_history: list[float] | None = None,
    ) -> float:
        """Decrease temperature by a constant amount each iteration."""
        _ = current_temperature, objective_value_history  # Not used in linear schedule
        return max(0.0, initial_temperature - self.alpha * iteration)


class ExponentialSchedule(TemperatureSchedule):
    """Exponential decay schedule."""

    def __init__(self, alpha: float) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in the range (0, 1) for exponential schedule.")
        self.alpha = alpha

    def get_temperature(
        self,
        initial_temperature: float,
        iteration: int,
        *,
        current_temperature: float | None = None,
        objective_value_history: list[float] | None = None,
    ) -> float:
        """Decrease temperature by a constant multiplicative factor."""
        _ = current_temperature, objective_value_history  # Not used in exponential schedule
        return initial_temperature * (self.alpha**iteration)


class LogarithmicSchedule(TemperatureSchedule):
    """Logarithmic decay schedule."""

    def __init__(self, c: float, d: float) -> None:
        self.c = c
        self.d = d

    def get_temperature(
        self,
        initial_temperature: float,
        iteration: int,
        *,
        current_temperature: float | None = None,
        objective_value_history: list[float] | None = None,
    ) -> float:
        """Decrease temperature according to a logarithmic schedule."""
        _ = initial_temperature, current_temperature, objective_value_history  # Not used in logarithmic schedule
        return self.c / math.log(iteration + self.d)


class AdaptiveSchedule(TemperatureSchedule):
    """Triki adaptive temperature schedule.

    Uses formula ``T_{k+1} = T_k * (1 - T_k * delta / sigma_sq)`` where
    ``sigma_sq`` is the variance of all objective values sampled so far and
    ``delta`` is a constant target decrease in cost per Metropolis chain.

    When ``delta`` is not provided, it is derived automatically on the first
    call that has sufficient objective value history as
    ``stdev(objective_value_history) / mu``, then held constant for the rest
    of the run.

    Falls back to current temperature when ``objective_value_history`` has
    fewer than 2 entries, when variance is zero, or when the factor
    ``T_k * delta / sigma_sq >= 1``.
    """

    def __init__(self, delta: float | None = None, mu: float = 5.0) -> None:
        self.delta = delta
        self.mu = mu

    def get_temperature(
        self,
        initial_temperature: float,
        iteration: int,
        *,
        current_temperature: float | None = None,
        objective_value_history: list[float] | None = None,
    ) -> float:
        """Decrease temperature adaptively based on spread of sampled objective values."""
        _ = iteration  # Not used in adaptive schedule
        t_k = current_temperature if current_temperature is not None else initial_temperature

        # Not enough data to adapt, return current temperature
        if objective_value_history is None or len(objective_value_history) < 2:
            return t_k

        sigma_sq = statistics.variance(objective_value_history)
        # No variation in objective values, keep temperature the same
        if sigma_sq == 0.0:
            return t_k

        # Derive delta from spread of objective values, scaled by mu, if not provided
        delta = self.delta if self.delta is not None else statistics.stdev(objective_value_history) / self.mu

        factor = t_k * delta / sigma_sq
        # Avoid negative or zero temperature, keep the same
        if factor >= 1.0:
            return t_k

        return t_k * (1 - factor)


def _validate_state_shape(
    state: Mapping[str, object],
    *,
    state_name: str,
    expected_keys: set[str] | None,
    state_validator: StateValidator | None,
) -> None:
    """Validate structural state requirements shared by initial and runtime states."""
    if not all(isinstance(k, str) for k in state):
        raise ValueError(f"All keys in {state_name} must be strings.")
    if expected_keys is not None and not expected_keys.issubset(state.keys()):
        missing_keys = expected_keys - state.keys()
        raise ValueError(f"{state_name} is missing expected keys: {missing_keys}")
    if state_validator is not None and not state_validator(state):
        raise ValueError(f"{state_name} failed validation by state_validator.")


def _validate_initial_state(
    initial_state: Mapping[str, object],
    constraints: list[ConstraintDelegate],
    expected_keys: set[str] | None,
    state_validator: StateValidator | None,
) -> None:
    """Validate initial state."""
    _validate_state_shape(
        initial_state,
        state_name="initial_state",
        expected_keys=expected_keys,
        state_validator=state_validator,
    )
    if constraints:
        violations = [i for i, c in enumerate(constraints) if not c(initial_state)]
        if violations:
            raise ValueError(f"initial_state violates constraints at indices: {violations}")


def _metropolis_acceptance(
    current_internal_score: float,
    neighbor_internal_score: float,
    temperature: float,
    rng: random.Random,
) -> bool:
    """Metropolis-Hastings acceptance criterion.

    Returns whether to accept the neighbor state based on internal score.

    Args:
        current_internal_score: Internal score of the current state.
        neighbor_internal_score: Internal score of the proposed neighbor state.
        temperature: Current temperature controlling acceptance probability.
        rng: Random number generator for stochastic acceptance.

    Returns:
        accepted: whether the neighbor state is accepted.
    """
    # Always accept better states
    if neighbor_internal_score < current_internal_score:
        return True

    # Never accept worse states if temperature is zero or negative
    if temperature <= 0:
        return False

    # If neighbor is worse, accept with probabilty exp(-delta / temperature)
    delta = neighbor_internal_score - current_internal_score
    acceptance_probability = math.exp(-delta / temperature)

    # Use seeded instance rather than global random for testability and reproducibility
    accepted = rng.random() < acceptance_probability
    return accepted


class SimulatedAnnealingPattern(Delegate):
    """General simulated annealing optimization pattern."""

    def __init__(
        self,
        *,
        neighbor_delegate: NeighborDelegate | None = None,
        modifications_delegate: ModificationsDelegate | None = None,
        objective_delegate: ObjectiveDelegate,
        objective_mode: Literal["minimize", "maximize"] = "minimize",
        constraints: list[ConstraintDelegate] | None = None,
        initial_state: Mapping[str, object] | None = None,
        initial_state_generator: InitialStateGenerator | None = None,
        expected_keys: set[str] | None = None,
        state_validator: StateValidator | None = None,
        initial_temperature: float = 100.0,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        convergence_steps: int = 5,
        # TODO: do we want to support user-defined temperature schedules?
        temperature_schedule: TemperatureSchedule | None = None,
        random_seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and validate baseline simulated annealing settings.

        Args:
            neighbor_delegate: Delegate that generates a neighboring solution given the current solution.
                Mutually exclusive with modifications_delegate. (Default: None)
            modifications_delegate: Delegate that returns list of possible modifications to current solution.
                Mutually exclusive with neighbor_delegate. (Default: None)
            objective_delegate: Delegate that computes the objective function value for a given solution.
            objective_mode: Whether to minimize or maximize the objective function. (Default: "minimize")
            constraints: Optional list of delegates that define constraints for the optimization. (Default: None)
            initial_state: Initial state for the optimization.
                Mutually exclusive with initial_state_generator. (Default: None)
            initial_state_generator: Callable that generates the initial state.
                Mutually exclusive with initial_state. (Default: None)
            expected_keys: Optional set of expected keys that must be present in initial state. (Default: None)
            state_validator: Optional callable that validates a state. (Default: None)
            initial_temperature: Starting temperature for the annealing process. (Default: 100.0)
            max_iterations: Maximum number of iterations to perform. (Default: 100)
            convergence_threshold: Minimum absolute change in objective value to consider non-converged. (Default: 1e-6)
            convergence_steps: Number of consecutive steps with objective value change below threshold. (Default: 5)
            temperature_schedule: Schedule for temperature decay. (Default: ExponentialSchedule)
            random_seed: Seed for random number generation. (Default: None)
            tracer: Optional tracer for workflow and debugging.
        """
        # Validate mutually exclusive delegates
        if neighbor_delegate is None and modifications_delegate is None:
            raise ValueError("Either neighbor_delegate or modifications_delegate must be provided.")
        if neighbor_delegate is not None and modifications_delegate is not None:
            raise ValueError("neighbor_delegate and modifications_delegate are mutually exclusive.")

        # Validate initial state exclusivity
        if initial_state is None and initial_state_generator is None:
            raise ValueError("Either initial_state or initial_state_generator must be provided.")
        if initial_state is not None and initial_state_generator is not None:
            raise ValueError("initial_state and initial_state_generator are mutually exclusive.")

        # Validate inputs
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if initial_temperature < 1:
            raise ValueError("initial_temperature must be >= 1.")
        if convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be > 0.")
        if convergence_steps < 1:
            raise ValueError("convergence_steps must be >= 1.")
        if initial_state is not None:
            _validate_initial_state(initial_state, constraints or [], expected_keys, state_validator)

        self._neighbor_delegate = neighbor_delegate
        self._modifications_delegate = modifications_delegate
        self._objective_delegate = objective_delegate
        self._objective_mode = objective_mode
        self._constraints = constraints or []
        self._initial_state = dict(initial_state) if initial_state is not None else None
        self._initial_state_generator = initial_state_generator
        self._expected_keys = expected_keys
        self._state_validator = state_validator
        self._initial_temperature = initial_temperature
        self._max_iterations = max_iterations
        self._temperature_schedule = temperature_schedule or ExponentialSchedule(alpha=0.95)
        self._random_seed = random_seed
        self.convergence_threshold = convergence_threshold
        self.convergence_steps = convergence_steps
        self._rng = random.Random(random_seed) if random_seed is not None else random.Random()
        self._tracer = tracer
        self.workflow: Workflow | None = None

    def _to_internal_score(self, objective_value: float) -> float:
        """Convert objective value to internal score for optimization."""
        return -objective_value if self._objective_mode == "maximize" else objective_value

    def run(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute the simulated annealing pattern."""
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
        """Compile one simulated annealing workflow."""
        run_context = resolve_pattern_run_context(
            prompt=prompt,
            default_request_id_prefix=None,
            default_dependencies={},
            request_id=request_id,
            dependencies=dependencies,
        )
        workflow = self._build_workflow(
            run_context.prompt,
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
        )
        return build_compiled_pattern_execution(
            workflow=workflow,
            pattern_name="SimulatedAnnealingPattern",
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            input_payload={
                **run_context.normalized_input,
                "mode": MODE_SIMULATED_ANNEALING,
                "objective_mode": self._objective_mode,
                "initial_temperature": self._initial_temperature,
                "max_iterations": self._max_iterations,
                "convergence_threshold": self.convergence_threshold,
                "convergence_steps": self.convergence_steps,
                "temperature_schedule": type(self._temperature_schedule).__name__,
            },
            workflow_request_id=f"{run_context.request_id}:simulated_annealing_workflow",
            finalize=lambda workflow_result: _build_simulated_annealing_result(
                workflow_result=workflow_result,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
                initial_state=self._initial_state or {},
                objective_mode=self._objective_mode,
                initial_temperature=self._initial_temperature,
                max_iterations=self._max_iterations,
                convergence_threshold=self.convergence_threshold,
                convergence_steps=self.convergence_steps,
                temperature_schedule_name=type(self._temperature_schedule).__name__,
                random_seed=self._random_seed,
            ),
        )

    def _build_workflow(
        self,
        prompt: str,
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> Workflow:
        """Build the workflow wrapper for one simulated annealing run."""

        def _get_initial_loop_state() -> dict[str, object]:
            if self._initial_state is not None:
                initial_state = self._initial_state
            else:
                assert self._initial_state_generator is not None
                initial_state = dict(self._initial_state_generator())
                _validate_initial_state(
                    initial_state,
                    self._constraints,
                    self._expected_keys,
                    self._state_validator,
                )
            initial_objective_value = self._objective_delegate(initial_state)
            return {
                "current_state": dict(initial_state),
                "current_objective_value": initial_objective_value,
                "best_state": dict(initial_state),
                "best_objective_value": initial_objective_value,
                "current_temperature": self._initial_temperature,
                "objective_value_history": [initial_objective_value],
                "iteration": 0,
                "should_continue": True,
                "convergence_counter": 0,
                "last_objective_value": initial_objective_value,
                "terminated_reason": None,
            }

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
            raw_loop_state = context.get("loop_state")
            loop_state = dict(raw_loop_state) if isinstance(raw_loop_state, Mapping) else {}
            iteration = int(loop_state.get("iteration") or 0)
            current_temperature = float(loop_state.get("current_temperature", self._initial_temperature))
            objective_value_history = list(loop_state.get("objective_value_history", []))

            # Generate temperature for this iteration
            temperature = self._temperature_schedule.get_temperature(
                self._initial_temperature,
                iteration,
                current_temperature=current_temperature,
                objective_value_history=objective_value_history,
            )

            # Generate neighbor
            if self._neighbor_delegate is not None:
                neighbor = self._neighbor_delegate(loop_state["current_state"])
            else:
                assert self._modifications_delegate is not None
                modifications = self._modifications_delegate(loop_state["current_state"])
                if not modifications:
                    raise ValueError("modifications_delegate must return at least one modification.")
                selected_modification = self._rng.choice(modifications)
                neighbor = {**loop_state["current_state"], **selected_modification}

            _validate_state_shape(
                neighbor,
                state_name="neighbor state",
                expected_keys=self._expected_keys,
                state_validator=self._state_validator,
            )

            # Invalid neighbors count as iterations, but do not update state
            if self._constraints and not all(c(neighbor) for c in self._constraints):
                return {
                    **loop_state,
                    "iteration": iteration + 1,
                    "current_temperature": temperature,
                    "objective_value_history": objective_value_history,
                }

            # Compute neighbor objective value
            neighbor_objective_value = self._objective_delegate(neighbor)

            # Determine whether to accept neighbor
            accepted = _metropolis_acceptance(
                current_internal_score=self._to_internal_score(loop_state["current_objective_value"]),
                neighbor_internal_score=self._to_internal_score(neighbor_objective_value),
                temperature=temperature,
                rng=self._rng,
            )

            # Update state and objective value based on acceptance
            current_state = neighbor if accepted else loop_state["current_state"]
            current_objective_value = neighbor_objective_value if accepted else loop_state["current_objective_value"]
            is_better = self._to_internal_score(current_objective_value) < self._to_internal_score(
                loop_state["best_objective_value"]
            )
            best_state = current_state if is_better else loop_state["best_state"]
            best_objective_value = current_objective_value if is_better else loop_state["best_objective_value"]
            objective_value_history = [*objective_value_history, current_objective_value]

            # Check for termination conditions
            terminated_reason = None
            should_continue = True

            # Determine if max iterations reached
            max_iterations_reached = (iteration + 1) >= self._max_iterations
            if max_iterations_reached:
                terminated_reason = "max_iterations_reached"
                should_continue = False

            # Determine if convergence reached
            convergence_counter = int(loop_state.get("convergence_counter", 0))
            last_objective_value = loop_state.get("last_objective_value", current_objective_value)
            if abs(current_objective_value - last_objective_value) < self.convergence_threshold:
                convergence_counter += 1
                if convergence_counter >= self.convergence_steps:
                    terminated_reason = "converged"
                    should_continue = False
            else:
                convergence_counter = 0

            return {
                "current_state": current_state,
                "current_objective_value": current_objective_value,
                "best_state": best_state,
                "best_objective_value": best_objective_value,
                "current_temperature": temperature,
                "objective_value_history": objective_value_history,
                "iteration": iteration + 1,
                "should_continue": should_continue,
                "convergence_counter": convergence_counter,
                "last_objective_value": current_objective_value,
                "terminated_reason": terminated_reason,
            }

        wrapped_handler = wrap_iteration_handler(
            _run_iteration,
            error_prefix="SimulatedAnnealingPattern iteration",
        )
        loop_callbacks = build_loop_callbacks(
            iteration_step_id="simulated_annealing_iteration",
            iteration_handler=wrapped_handler,
        )

        workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
            steps=[
                LoopStep(
                    step_id="simulated_annealing",
                    steps=(
                        LogicStep(
                            step_id="simulated_annealing_iteration",
                            handler=loop_callbacks.iteration_handler,
                        ),
                    ),
                    max_iterations=self._max_iterations,
                    initial_state=_get_initial_loop_state(),
                    continue_predicate=loop_callbacks.continue_predicate,
                    state_reducer=loop_callbacks.state_reducer,
                    execution_mode="sequential",
                    failure_policy="propagate_failed_state",
                )
            ],
        )
        self.workflow = workflow
        return workflow


def _build_simulated_annealing_result(
    *,
    workflow_result: ExecutionResult,
    request_id: str,
    dependencies: Mapping[str, object],
    initial_state: Mapping[str, object],
    objective_mode: Literal["minimize", "maximize"],
    initial_temperature: float,
    max_iterations: int,
    convergence_threshold: float,
    convergence_steps: int,
    temperature_schedule_name: str,
    random_seed: int | None,
) -> ExecutionResult:
    """Build the final result from one simulated annealing workflow execution."""
    loop_step_result = workflow_result.step_results.get("simulated_annealing")
    loop_output = dict(loop_step_result.output) if loop_step_result is not None else {}
    final_state_raw = loop_output.get("final_state")
    final_state = dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}
    workflow_artifacts = workflow_result.output.get("artifacts", [])
    terminated_reason = str(
        final_state.get("terminated_reason", loop_output.get("terminated_reason"))
        if workflow_result.success
        else "workflow_failure"
    )
    return build_pattern_execution_result(
        success=workflow_result.success,
        final_output={
            "best_state": final_state.get("best_state"),
            "best_objective_value": final_state.get("best_objective_value"),
            "iterations": final_state.get("iteration"),
        },
        terminated_reason=terminated_reason,
        details={
            "initial_state": dict(initial_state),
            "objective_mode": objective_mode,
            "initial_temperature": initial_temperature,
            "max_iterations": max_iterations,
            "convergence_threshold": convergence_threshold,
            "convergence_steps": convergence_steps,
            "temperature_schedule": temperature_schedule_name,
            "current_state": final_state.get("current_state"),
            "current_objective_value": final_state.get("current_objective_value"),
        },
        workflow_payload=workflow_result.to_dict(),
        artifacts=workflow_artifacts,
        request_id=request_id,
        dependencies=dependencies,
        mode=MODE_SIMULATED_ANNEALING,
        metadata={
            "objective_mode": objective_mode,
            "initial_temperature": initial_temperature,
            "max_iterations": max_iterations,
            "temperature_schedule": temperature_schedule_name,
            "convergence_threshold": convergence_threshold,
            "convergence_steps": convergence_steps,
            "random_seed": random_seed,
        },
        requested_mode=MODE_SIMULATED_ANNEALING,
        resolved_mode=MODE_SIMULATED_ANNEALING,
    )


__all__ = [
    "AdaptiveSchedule",
    "ConstraintDelegate",
    "ExponentialSchedule",
    "InitialStateGenerator",
    "LinearSchedule",
    "LogarithmicSchedule",
    "ModificationsDelegate",
    "NeighborDelegate",
    "ObjectiveDelegate",
    "SimulatedAnnealingPattern",
    "StateValidator",
    "TemperatureSchedule",
]
