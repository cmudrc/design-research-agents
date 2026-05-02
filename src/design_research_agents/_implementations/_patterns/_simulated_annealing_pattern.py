"""Reusable ``simulated_annealing`` orchestration scaffold."""

from __future__ import annotations

import math
import random
import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

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
ObjectiveDelegate = Callable[[Mapping[str, object]], float]
ConstraintDelegate = Callable[[Mapping[str, object]], bool]


class TemperatureSchedule(ABC):
    """Base class for temperature decay schedules."""

    @abstractmethod
    def get_temperature(
        self,
        initial_temperature: float,
        iteration: int,
        *,
        current_temperature: float | None = None,
        energy_history: list[float] | None = None,
    ) -> float:
        """Return the temperature for one iteration.

        Args:
            initial_temperature: The initial temperature configured for the SA run.
            iteration: The current iteration number (starting from 0).
            current_temperature: The temperature from the previous iteration, if applicable.
            energy_history: List of objective values from previous iterations, if applicable.

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
        energy_history: list[float] | None = None,
    ) -> float:
        """Decrease temperature by a constant amount each iteration."""
        _ = current_temperature, energy_history  # Not used in linear schedule
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
        energy_history: list[float] | None = None,
    ) -> float:
        """Decrease temperature by a constant multiplicative factor."""
        _ = current_temperature, energy_history  # Not used in exponential schedule
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
        energy_history: list[float] | None = None,
    ) -> float:
        """Decrease temperature according to a logarithmic schedule."""
        _ = initial_temperature, current_temperature, energy_history  # Not used in logarithmic schedule
        return self.c / math.log(iteration + self.d)


class AdaptiveSchedule(TemperatureSchedule):
    """Triki adaptive temperature schedule.

    Uses formula ``T_{k+1} = T_k * (1 - T_k * delta / sigma_sq)`` where
    ``sigma_sq`` is the variance of all objective values sampled so far and
    ``delta`` is a constant target decrease in cost per Metropolis chain.

    When ``delta`` is not provided, it is derived automatically on the first
    call that has sufficient energy history as ``stdev(energy_history) / mu``,
    then held constant for the rest of the run.

    Falls back to current temperature when ``energy_history`` has fewer
    than 2 entries, when variance is zero, or when the factor
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
        energy_history: list[float] | None = None,
    ) -> float:
        """Decrease temperature adaptively based on spread of sampled objective values."""
        _ = iteration  # Not used in adaptive schedule
        t_k = current_temperature if current_temperature is not None else initial_temperature

        # Not enough data to adapt, return current temperature
        if energy_history is None or len(energy_history) < 2:
            return t_k

        sigma_sq = statistics.variance(energy_history)
        # No variation in energy, keep temperature the same
        if sigma_sq == 0.0:
            return t_k

        # Derive delta from spread of energy values, scaled by mu, if not provided
        delta = self.delta if self.delta is not None else statistics.stdev(energy_history) / self.mu

        factor = t_k * delta / sigma_sq
        # Avoid negative or zero temperature, keep the same
        if factor >= 1.0:
            return t_k

        return t_k * (1 - factor)


def _metropolis_acceptance(
    current_energy: float,
    neighbor_energy: float,
    temperature: float,
    rng: random.Random,
) -> bool:
    """Metropolis-Hastings acceptance criterion.

    Returns whether to accept the neighbor state.

    Args:
        current_energy: Energy of the current state.
        neighbor_energy: Energy of the proposed neighbor state.
        temperature: Current temperature controlling acceptance probability.
        rng: Random number generator for stochastic acceptance.

    Returns:
        accepted: whether the neighbor state is accepted.
    """
    # Always accept better states
    if neighbor_energy < current_energy:
        return True

    # Never accept worse states if temperature is zero or negative
    if temperature <= 0:
        return False

    # If neighbor is worse, accept with probabilty exp(-delta / temperature)
    delta = neighbor_energy - current_energy
    acceptance_probability = math.exp(-delta / temperature)

    # Use seeded instance rather than global random for testability and reproducibility
    accepted = rng.random() < acceptance_probability
    return accepted


class SimulatedAnnealingPattern(Delegate):
    """General simulated annealing optimization pattern."""

    # TODO: should energy delegate just be an objective function?
    # TODO: should neighbor delegate just be a list of possible modifications
    # from current state, rather than a function that generates one neighbor?
    def __init__(
        self,
        *,
        neighbor_delegate: NeighborDelegate,
        objective_delegate: ObjectiveDelegate,
        constraints: list[ConstraintDelegate] | None = None,
        initial_state: Mapping[str, object],
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
            objective_delegate: Delegate that computes the objective function value for a given solution.
            constraints: Optional list of delegates that define constraints for the optimization. (Default: None)
            initial_state: Initial state for the optimization.
            initial_temperature: Starting temperature for the annealing process. (Default: 100.0)
            max_iterations: Maximum number of iterations to perform. (Default: 100)
            convergence_threshold: Minimum absolute change in objective value to consider non-converged. (Default: 1e-6)
            convergence_steps: Number of consecutive steps with objective value change below threshold. (Default: 5)
            temperature_schedule: Schedule for temperature decay. (Default: ExponentialSchedule)
            random_seed: Seed for random number generation. (Default: None)
            tracer: Optional tracer for workflow and debugging.
        """
        # Validate inputs
        if not all(isinstance(k, str) for k in initial_state):
            raise ValueError("All keys in initial_state must be strings.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if initial_temperature < 1:
            raise ValueError("initial_temperature must be >= 1.")
        if convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be > 0.")
        if convergence_steps < 1:
            raise ValueError("convergence_steps must be >= 1.")
        if constraints:
            violations = [i for i, c in enumerate(constraints) if not c(initial_state)]
            if violations:
                raise ValueError("initial_state must not violate constraints.")

        self._neighbor_delegate = neighbor_delegate
        self._objective_delegate = objective_delegate
        self._constraints = constraints or []
        self._initial_state = dict(initial_state)
        self._initial_temperature = initial_temperature
        self._max_iterations = max_iterations
        self._temperature_schedule = temperature_schedule or ExponentialSchedule(alpha=0.95)
        self._random_seed = random_seed
        self.convergence_threshold = convergence_threshold
        self.convergence_steps = convergence_steps
        self._rng = random.Random(random_seed) if random_seed is not None else random.Random()
        self._tracer = tracer
        self.workflow: Workflow | None = None

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
                initial_state=self._initial_state,
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
            initial_energy = self._objective_delegate(self._initial_state)
            return {
                "current_state": dict(self._initial_state),
                "current_energy": initial_energy,
                "best_state": dict(self._initial_state),
                "best_energy": initial_energy,
                "current_temperature": self._initial_temperature,
                "energy_history": [initial_energy],
                "iteration": 0,
                "should_continue": True,
                "convergence_counter": 0,
                "last_energy": initial_energy,
                "terminated_reason": None,
            }

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
            raw_loop_state = context.get("loop_state")
            loop_state = dict(raw_loop_state) if isinstance(raw_loop_state, Mapping) else {}
            iteration = int(loop_state.get("iteration") or 0)
            current_temperature = float(loop_state.get("current_temperature", self._initial_temperature))
            energy_history = list(loop_state.get("energy_history", []))

            # Generate temperature for this iteration
            temperature = self._temperature_schedule.get_temperature(
                self._initial_temperature,
                iteration,
                current_temperature=current_temperature,
                energy_history=energy_history,
            )

            # Generate neighbor
            neighbor = self._neighbor_delegate(loop_state["current_state"])

            # Invalid neighbors count as iterations, but do not update state
            if self._constraints and not all(c(neighbor) for c in self._constraints):
                return {
                    **loop_state,
                    "iteration": iteration + 1,
                    "current_temperature": temperature,
                    "energy_history": energy_history,
                }

            # Compute neighbor energy
            neighbor_energy = self._objective_delegate(neighbor)

            # Determine whether to accept neighbor
            accepted = _metropolis_acceptance(
                current_energy=loop_state["current_energy"],
                neighbor_energy=neighbor_energy,
                temperature=temperature,
                rng=self._rng,
            )

            # Update state and energy based on acceptance
            current_state = neighbor if accepted else loop_state["current_state"]
            current_energy = neighbor_energy if accepted else loop_state["current_energy"]
            best_state = current_state if current_energy < loop_state["best_energy"] else loop_state["best_state"]
            best_energy = min(current_energy, loop_state["best_energy"])
            energy_history = [*energy_history, current_energy]

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
            last_energy = loop_state.get("last_energy", current_energy)
            if abs(current_energy - last_energy) < self.convergence_threshold:
                convergence_counter += 1
                if convergence_counter >= self.convergence_steps:
                    terminated_reason = "converged"
                    should_continue = False
            else:
                convergence_counter = 0

            return {
                "current_state": current_state,
                "current_energy": current_energy,
                "best_state": best_state,
                "best_energy": best_energy,
                "current_temperature": temperature,
                "energy_history": energy_history,
                "iteration": iteration + 1,
                "should_continue": should_continue,
                "convergence_counter": convergence_counter,
                "last_energy": current_energy,
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
    initial_temperature: float,
    max_iterations: int,
    convergence_threshold: float,
    convergence_steps: int,
    temperature_schedule_name: str,
    random_seed: int | None,
) -> ExecutionResult:
    """Build the final result from one simulated annealing workflow execution."""
    loop_step_result = workflow_result.step_results.get("simulated_annealing_loop")
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
            "best_energy": final_state.get("best_energy"),
            "iterations": final_state.get("iteration"),
        },
        terminated_reason=terminated_reason,
        details={
            "initial_state": dict(initial_state),
            "initial_temperature": initial_temperature,
            "max_iterations": max_iterations,
            "convergence_threshold": convergence_threshold,
            "convergence_steps": convergence_steps,
            "temperature_schedule": temperature_schedule_name,
            "current_state": final_state.get("current_state"),
            "current_energy": final_state.get("current_energy"),
        },
        workflow_payload=workflow_result.to_dict(),
        artifacts=workflow_artifacts,
        request_id=request_id,
        dependencies=dependencies,
        mode=MODE_SIMULATED_ANNEALING,
        metadata={
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
    "LinearSchedule",
    "LogarithmicSchedule",
    "NeighborDelegate",
    "ObjectiveDelegate",
    "SimulatedAnnealingPattern",
    "TemperatureSchedule",
]
