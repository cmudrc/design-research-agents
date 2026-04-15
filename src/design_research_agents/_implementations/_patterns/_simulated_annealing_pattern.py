"""Reusable ``simulated_annealing`` orchestration scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import random
from collections.abc import Callable, Mapping

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._workflow import DelegateTarget, LogicStep, LoopStep
from design_research_agents._runtime._patterns import (
    MODE_SIMULATED_ANNEALING,
    build_compiled_pattern_execution,
    build_loop_callbacks,
    build_pattern_execution_result,
    resolve_pattern_run_context,
    wrap_iteration_handler
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution, Workflow

NeighborDelegate = Callable[[Mapping[str, object]], Mapping[str, object]]
EnergyDelegate = Callable[[Mapping[str, object]], float]


class TemperatureSchedule(ABC):
    """Base class for temperature decay schedules."""

    @abstractmethod
    def get_temperature(self, initial_temperature: float, iteration: int) -> float:
        """Return the temperature for one iteration."""


class LinearSchedule(TemperatureSchedule):
    """Linear decay schedule."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def get_temperature(self, initial_temperature: float, iteration: int) -> float:
        """Decrease temperature by a constant amount each iteration."""
        return max(0.0, initial_temperature - self.alpha * iteration)


class ExponentialSchedule(TemperatureSchedule):
    """Exponential decay schedule."""

    def __init__(self, alpha: float) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in the range (0, 1) for exponential schedule.")
        self.alpha = alpha

    def get_temperature(self, initial_temperature: float, iteration: int) -> float:
        """Decrease temperature by a constant multiplicative factor."""
        return initial_temperature * (self.alpha**iteration)


class LogarithmicSchedule(TemperatureSchedule):
    """Logarithmic decay schedule."""

    def __init__(self, c: float, d: float) -> None:
        self.c = c
        self.d = d

    def get_temperature(self, initial_temperature: float, iteration: int) -> float:
        """Decrease temperature according to a logarithmic schedule."""
        _ = initial_temperature  # log schedule depends only on c, d, and iteration
        return self.c / math.log(iteration + self.d)


def _metropolis_acceptance(
        current_energy: float,
        neighbor_energy: float,
        temperature: float, 
        rng: random.Random,
) -> tuple[bool, float]:
    """
    Metropolis-Hastings acceptance criterion that returns whether to accept the neighbor 
    and the acceptance probability.
    
    Args:
        current_energy: Energy of the current state.
        neighbor_energy: Energy of the proposed neighbor state.
        temperature: Current temperature controlling acceptance probability.
        rng: Random number generator for stochastic acceptance.
        
    Returns:
        Tuple of (accepted, acceptance_probability) where accepted is a boolean indicating
        whether the neighbor state is accepted, and acceptance_probability is the computed 
        probability of acceptance.
    """
    
    if neighbor_energy < current_energy:
        return True, 1.0  # Always accept better states
    if temperature <= 0:
        return False, 0.0  # Never accept worse states if temperature is zero or negative
    # If neighbor is worse, accept with probabilty exp(-delta / temperature)
    delta = neighbor_energy - current_energy
    acceptance_probability = math.exp(-delta / temperature)
    # Use seeded instance rather than global random for testability and reproducibility
    accepted = rng.random() < acceptance_probability
    return accepted, acceptance_probability
    

class SimulatedAnnealingPattern(Delegate):
    """General simulated annealing optimization pattern."""

    def __init__(
        self,
        *,
        neighbor_delegate: NeighborDelegate | DelegateTarget,
        energy_delegate: EnergyDelegate | DelegateTarget,
        initial_state: Mapping[str, object],
        initial_temperature: float = 100.0,
        max_iterations: int = 100,
        temperature_schedule: TemperatureSchedule | None = None,
        random_seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and validate baseline simulated annealing settings."""
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if initial_temperature < 1:
            raise ValueError("initial_temperature must be >= 1.")

        self._neighbor_delegate = neighbor_delegate
        self._energy_delegate = energy_delegate
        self._initial_state = dict(initial_state)
        self._initial_temperature = initial_temperature
        self._max_iterations = max_iterations
        self._temperature_schedule = temperature_schedule or ExponentialSchedule(alpha=0.95)
        self._random_seed = random_seed
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
            initial_energy = self._energy_delegate(self._initial_state)
            return {
                "current_state": dict(self._initial_state),
                "current_energy": initial_energy,
                "best_state": dict(self._initial_state),
                "best_energy": initial_energy,
                "iteration": 0,
                "should_continue": True,
            }

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
            loop_state = dict(context.get("loop_state"))
            iteration = int(loop_state.get("iteration"))

            temperature = self._temperature_schedule.get_temperature(
                self._initial_temperature, iteration
            )

            neighbor = self._neighbor_delegate(loop_state["current_state"])
            neighbor_energy = self._energy_delegate(neighbor)

            accepted, _ = _metropolis_acceptance(
                current_energy=loop_state["current_energy"],
                neighbor_energy=neighbor_energy,
                temperature=temperature,
                rng=self._rng,
            )

            current_state = neighbor if accepted else loop_state["current_state"]
            current_energy = neighbor_energy if accepted else loop_state["current_energy"]

            best_state = current_state if current_energy < loop_state["best_energy"] else loop_state["best_state"]
            best_energy = min(current_energy, loop_state["best_energy"])

            return {
                "current_state": current_state,
                "current_energy": current_energy,
                "best_state": best_state,
                "best_energy": best_energy,
                "iteration": iteration + 1,
                "should_continue": (iteration + 1) < self._max_iterations,
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
                        )
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
    temperature_schedule_name: str,
    random_seed: int | None,
) -> ExecutionResult:
    """Build the final result from one simulated annealing workflow execution."""
    step_result = workflow_result.step_results.get("simulated_annealing_loop")
    step_output = dict(step_result.output) if step_result is not None else {}
    workflow_artifacts = workflow_result.output.get("artifacts", [])
    terminated_reason = "max_iterations_reached" if workflow_result.success else "workflow_failure"
    return build_pattern_execution_result(
        success=workflow_result.success,
        final_output={
            "best_state": step_output.get("best_state"),
            "best_energy": step_output.get("best_energy"),
            "iterations": step_output.get("iteration"),
        },
        terminated_reason=terminated_reason,
        details={
            "initial_state": dict(initial_state),
            "initial_temperature": initial_temperature,
            "max_iterations": max_iterations,
            "temperature_schedule": temperature_schedule_name,
            "current_state": step_output.get("current_state"),
            "current_energy": step_output.get("current_energy"),
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
            "random_seed": random_seed,
        },
        requested_mode=MODE_SIMULATED_ANNEALING,
        resolved_mode=MODE_SIMULATED_ANNEALING,
    )


__all__ = [
    "EnergyDelegate",
    "ExponentialSchedule",
    "LinearSchedule",
    "LogarithmicSchedule",
    "NeighborDelegate",
    "SimulatedAnnealingPattern",
    "TemperatureSchedule",
]
