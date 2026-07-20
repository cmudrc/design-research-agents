"""Reusable ``reinforcement learning`` orchestration chunk."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol, cast

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._workflow import LogicStep, LoopStep
from design_research_agents._runtime._patterns import (
    MODE_REINFORCEMENT_LEARNING,
    build_compiled_pattern_execution,
    build_loop_callbacks,
    build_pattern_execution_result,
    build_workflow_output_payload,
    resolve_pattern_run_context,
    wrap_iteration_handler,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution, Workflow

# Design state
RLState = Mapping[str, object]
# Discrete action name or continuous parameter dict
RLAction = str | Mapping[str, object]
# Legacy episode experience retained for existing custom policies
Trajectory = list[tuple[RLState, RLAction, float]]

# Returns initial state for new episode
EnvironmentResetDelegate = Callable[[], RLState]
# Legacy ``done`` result or Gymnasium-style terminated/truncated result
EnvironmentStepResult = tuple[RLState, float, bool] | tuple[RLState, float, bool, bool, Mapping[str, object]]
EnvironmentStepDelegate = Callable[[RLState, RLAction], EnvironmentStepResult]
# Maps an arbitrary design state to one stable tabular key
StateKeyDelegate = Callable[[RLState], str]
EvaluationActionDelegate = Callable[[RLState], RLAction]

_GLOBAL_STATE_KEY = "__global__"
_TRACE_MAX_DEPTH = 8
_TRACE_MAX_ITEMS = 100
_TRACE_MAX_TEXT_LENGTH = 2_000
_TRACE_DETAILS = frozenset({"summary", "transitions", "full"})


@dataclass(frozen=True, slots=True)
class RLTransition:
    """One normalized agent-environment transition.

    The pattern accepts the original three-value environment result for
    compatibility, but records every interaction with explicit termination and
    truncation semantics. Transition-aware custom policies receive this richer
    contract through ``observe_transition(...)``.

    Attributes:
        state: Observation before the action.
        action: Action selected by the policy.
        reward: Finite scalar reward returned by the environment.
        next_state: Observation after the action.
        terminated: Whether the environment reached an MDP terminal state.
        truncated: Whether an external limit ended the episode.
        info: Auxiliary metrics or diagnostic information from the environment.
    """

    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    terminated: bool
    truncated: bool = False
    info: Mapping[str, object] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Return whether either terminal condition ended the episode."""
        return self.terminated or self.truncated

    def as_legacy_tuple(self) -> tuple[RLState, RLAction, float]:
        """Return the historical ``(state, action, reward)`` policy input."""
        return self.state, self.action, self.reward


def _trace_snapshot(value: object, *, _depth: int = 0) -> object:
    """Return a bounded JSON-safe snapshot without affecting execution values."""
    if _depth >= _TRACE_MAX_DEPTH:
        return f"<max-depth:{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, str):
        if len(value) <= _TRACE_MAX_TEXT_LENGTH:
            return value
        return f"{value[:_TRACE_MAX_TEXT_LENGTH]}<truncated:{len(value) - _TRACE_MAX_TEXT_LENGTH}>"
    if isinstance(value, Mapping):
        return _trace_mapping_snapshot(value, depth=_depth)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _trace_sequence_snapshot(value, depth=_depth)
    return _trace_fallback_snapshot(value, depth=_depth)


def _trace_mapping_snapshot(value: Mapping[object, object], *, depth: int) -> dict[str, object]:
    """Return a bounded trace snapshot for one mapping."""
    items = list(value.items())
    snapshot = {str(key): _trace_snapshot(item, _depth=depth + 1) for key, item in items[:_TRACE_MAX_ITEMS]}
    if len(items) > _TRACE_MAX_ITEMS:
        snapshot["__truncated_items__"] = len(items) - _TRACE_MAX_ITEMS
    return snapshot


def _trace_sequence_snapshot(value: Sequence[object], *, depth: int) -> list[object]:
    """Return a bounded trace snapshot for one non-string sequence."""
    items = list(value)
    snapshot = [_trace_snapshot(item, _depth=depth + 1) for item in items[:_TRACE_MAX_ITEMS]]
    if len(items) > _TRACE_MAX_ITEMS:
        snapshot.append(f"<truncated-items:{len(items) - _TRACE_MAX_ITEMS}>")
    return snapshot


def _trace_fallback_snapshot(value: object, *, depth: int) -> object:
    """Best-effort normalization for array-like and opaque trace values."""
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _trace_snapshot(to_list(), _depth=depth + 1)
        except Exception:  # pragma: no cover - defensive logging boundary
            pass
    try:
        rendered = repr(value)
    except Exception:  # pragma: no cover - defensive logging boundary
        rendered = f"<{type(value).__name__}>"
    return _trace_snapshot(rendered, _depth=depth + 1)


def _copy_experience_value(value: object, *, name: str) -> object:
    """Copy one policy experience value so later mutation cannot rewrite it."""
    try:
        return deepcopy(value)
    except Exception as exc:
        raise TypeError(f"{name} must support defensive copying for trajectory capture.") from exc


def _normalize_state(value: object, *, name: str) -> dict[str, object]:
    """Validate and normalize one environment state mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings.")
    return dict(value)


@dataclass(frozen=True, slots=True)
class _RewardStabilityCriterion:
    """Private early-stop heuristic for explicitly requested reward stability."""

    threshold: float
    stable_episodes: int

    def observe(self, rewards: Sequence[float], previous_count: int) -> tuple[int, bool]:
        """Return the updated stable-change count and whether the criterion fired."""
        if len(rewards) < 2:
            return previous_count, False
        stable = abs(rewards[-1] - rewards[-2]) <= self.threshold
        current_count = previous_count + 1 if stable else 0
        return current_count, current_count >= self.stable_episodes


class RLPolicy(Protocol):
    """Legacy-compatible protocol for episode-updated RL policies."""

    def select_action(self, state: RLState) -> RLAction:
        """Select an action for the current state."""
        ...

    def update(self, trajectory: Trajectory) -> dict[str, object]:
        """Update the policy from one episode and return JSON-safe statistics."""
        ...

    def get_params(self) -> dict[str, object]:
        """Return lightweight JSON-safe policy state or a checkpoint reference."""
        ...


class TransitionRLPolicy(Protocol):
    """Optional protocol for policies that learn after each transition."""

    def select_action(self, state: RLState) -> RLAction:
        """Select an action for the current state."""
        ...

    def observe_transition(self, transition: RLTransition) -> None:
        """Observe and optionally learn from one completed transition."""
        ...

    def end_episode(self, transitions: Sequence[RLTransition]) -> dict[str, object]:
        """Finish one episode and return JSON-safe update statistics."""
        ...

    def get_params(self) -> dict[str, object]:
        """Return lightweight JSON-safe policy state or a checkpoint reference."""
        ...


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy with global or tabular Monte Carlo updates.

    Without ``state_key``, the policy estimates one global value per action.
    With ``state_key``, it estimates one value per state-action pair. Both modes
    are intended for small discrete action spaces; use a custom ``RLPolicy`` for
    function approximation or continuous actions.
    """

    def __init__(
        self,
        *,
        actions: Sequence[str],
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.0,
        gamma: float = 0.99,
        state_key: StateKeyDelegate | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize a global or tabular action-value policy.

        Args:
            actions: Unique, non-empty discrete action names.
            epsilon: Initial probability of selecting a random action.
            epsilon_decay: Multiplicative epsilon decay after each episode.
            epsilon_min: Lower bound applied when epsilon decays.
            gamma: Discount factor used for Monte Carlo returns.
            state_key: Optional callable mapping states to stable table keys.
            rng: Optional random-number generator for reproducible selection.

        Raises:
            ValueError: If an action or numeric policy setting is invalid.
        """
        normalized_actions = tuple(actions)
        if not normalized_actions:
            raise ValueError("actions must be a non-empty sequence.")
        if any(not isinstance(action, str) or not action.strip() for action in normalized_actions):
            raise ValueError("actions must contain non-empty strings.")
        if len(set(normalized_actions)) != len(normalized_actions):
            raise ValueError("actions must be unique.")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0 <= epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in [0, 1].")
        if not 0 <= epsilon_min <= 1:
            raise ValueError("epsilon_min must be in [0, 1].")
        if epsilon_min > epsilon:
            raise ValueError("epsilon_min must be <= epsilon.")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")

        self._actions = normalized_actions
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._gamma = gamma
        self._state_key = state_key
        self._rng = rng or random.Random()
        self._values: dict[str, dict[str, float]] = {}
        self._counts: dict[str, dict[str, int]] = {}
        if state_key is None:
            self._initialize_state(_GLOBAL_STATE_KEY)

    @property
    def epsilon(self) -> float:
        """Return the current exploration probability."""
        return self._epsilon

    def select_action(self, state: RLState) -> str:
        """Select a random action with epsilon probability, else a best action."""
        state_values, _ = self._tables_for_state(state)
        if self._rng.random() < self._epsilon:
            return self._rng.choice(self._actions)
        best_value = max(state_values.values())
        best_actions = [action for action, value in state_values.items() if value == best_value]
        return self._rng.choice(best_actions)

    def select_evaluation_action(self, state: RLState) -> str:
        """Return a deterministic greedy action without changing exploration state."""
        state_values = self._values.get(self._resolve_state_key(state))
        if state_values is None:
            return self._actions[0]
        best_value = max(state_values.values())
        return next(action for action in self._actions if state_values[action] == best_value)

    def update(self, trajectory: Trajectory) -> dict[str, object]:
        """Compute discounted episode returns and update configured value tables.

        This implementation uses first-visit Monte Carlo episode returns.
        When the same state-action pair recurs within one episode, only the return
        following its first occurrence updates the estimate.
        """
        if not trajectory:
            return {"mean_return": 0.0}

        n = len(trajectory)
        returns = [0.0] * n
        returns[-1] = trajectory[-1][2]
        for t in range(n - 2, -1, -1):
            returns[t] = trajectory[t][2] + self._gamma * returns[t + 1]

        first_visits: list[tuple[str, str, float]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for index, (state, action, _) in enumerate(trajectory):
            if not isinstance(action, str) or action not in self._actions:
                raise ValueError("trajectory contains an action not configured for this policy.")
            pair = (self._resolve_state_key(state), action)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            first_visits.append((pair[0], pair[1], returns[index]))

        for state_key, action, observed_return in first_visits:
            state_values, state_counts = self._tables_for_key(state_key)
            state_counts[action] += 1
            count = state_counts[action]
            state_values[action] += (observed_return - state_values[action]) / count

        self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)

        return {
            "mean_return": sum(returns) / len(returns),
            "epsilon": self._epsilon,
            "value_mode": self.value_mode,
        }

    def get_params(self) -> dict[str, object]:
        """Return a fresh snapshot of learned values and exploration state."""
        if self._state_key is None:
            return {
                "value_mode": self.value_mode,
                "action_values": dict(self._values[_GLOBAL_STATE_KEY]),
                "action_counts": dict(self._counts[_GLOBAL_STATE_KEY]),
                "epsilon": self._epsilon,
            }
        return {
            "value_mode": self.value_mode,
            "q_values": {key: dict(values) for key, values in self._values.items()},
            "state_action_counts": {key: dict(counts) for key, counts in self._counts.items()},
            "epsilon": self._epsilon,
        }

    @property
    def value_mode(self) -> str:
        """Return the configured global-action or state-action value mode."""
        return "global_action" if self._state_key is None else "state_action"

    def _resolve_state_key(self, state: RLState) -> str:
        """Return the global key or validate one user-provided state key."""
        if self._state_key is None:
            return _GLOBAL_STATE_KEY
        state_key = self._state_key(state)
        if not isinstance(state_key, str) or not state_key.strip():
            raise ValueError("state_key must return a non-empty string.")
        return state_key

    def _initialize_state(self, state_key: str) -> None:
        """Initialize zero-valued action tables for one previously unseen state."""
        self._values[state_key] = {action: 0.0 for action in self._actions}
        self._counts[state_key] = {action: 0 for action in self._actions}

    def _tables_for_key(self, state_key: str) -> tuple[dict[str, float], dict[str, int]]:
        """Return initialized value and count tables for one state key."""
        if state_key not in self._values:
            self._initialize_state(state_key)
        return self._values[state_key], self._counts[state_key]

    def _tables_for_state(self, state: RLState) -> tuple[dict[str, float], dict[str, int]]:
        """Return initialized value and count tables for one state."""
        return self._tables_for_key(self._resolve_state_key(state))


class ReinforcementLearningPattern(Delegate):
    """Reinforcement learning pattern with episodic agent-environment loop."""

    def __init__(
        self,
        *,
        environment_reset: EnvironmentResetDelegate,
        environment_step: EnvironmentStepDelegate,
        policy: RLPolicy | TransitionRLPolicy | None = None,
        actions: Sequence[str] | None = None,
        max_episodes: int = 100,
        max_steps_per_episode: int = 50,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.0,
        state_key: StateKeyDelegate | None = None,
        convergence_threshold: float | None = None,
        convergence_episodes: int = 5,
        trace_detail: str = "transitions",
        random_seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store the environment, policy, and bounded training configuration.

        Args:
            environment_reset: Callable returning the initial state for each episode.
            environment_step: Callable applying one action and returning either
                ``(next_state, reward, done)`` or Gymnasium-style
                ``(next_state, reward, terminated, truncated, info)``.
            policy: Custom structural policy implementation. Mutually exclusive with
                ``actions``.
            actions: Discrete action names used to build the default epsilon-greedy
                policy. Mutually exclusive with ``policy``.
            max_episodes: Maximum number of training episodes.
            max_steps_per_episode: Maximum environment transitions per episode.
            gamma: Discount factor for the default policy.
            epsilon: Initial exploration probability for the default policy.
            epsilon_decay: Per-episode exploration decay for the default policy.
            epsilon_min: Minimum exploration probability for the default policy.
            state_key: Optional state-to-string mapping that enables tabular
                state-action values for the default policy.
            convergence_threshold: Optional maximum reward change counted as
                stable. ``None`` disables the private reward-stability heuristic.
            convergence_episodes: Consecutive stable reward changes required to
                stop early when ``convergence_threshold`` is configured.
            trace_detail: Episode trace detail: ``"summary"`` stores episode
                metrics, ``"transitions"`` also stores step transitions, and
                ``"full"`` additionally stores a policy snapshot per episode.
                The final policy state is always retained.
            random_seed: Optional seed for default-policy action selection.
            tracer: Optional workflow tracer.

        Raises:
            ValueError: If the training bounds or policy selection are invalid.
        """
        if isinstance(max_episodes, bool) or not isinstance(max_episodes, int) or max_episodes < 1:
            raise ValueError("max_episodes must be >= 1.")
        if (
            isinstance(max_steps_per_episode, bool)
            or not isinstance(max_steps_per_episode, int)
            or max_steps_per_episode < 1
        ):
            raise ValueError("max_steps_per_episode must be >= 1.")
        if not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1].")
        if convergence_threshold is not None and (
            not math.isfinite(convergence_threshold) or convergence_threshold < 0
        ):
            raise ValueError("convergence_threshold must be finite and >= 0.")
        if (
            isinstance(convergence_episodes, bool)
            or not isinstance(convergence_episodes, int)
            or convergence_episodes < 1
        ):
            raise ValueError("convergence_episodes must be >= 1.")
        if trace_detail not in _TRACE_DETAILS:
            raise ValueError(f"trace_detail must be one of {sorted(_TRACE_DETAILS)}.")
        if policy is None and not actions:
            if state_key is not None:
                raise ValueError("state_key requires actions when no custom policy is provided.")
            raise ValueError("Either policy or actions must be provided.")
        if policy is not None and (actions is not None or state_key is not None):
            raise ValueError("policy is mutually exclusive with actions and state_key.")

        observes_transitions = callable(getattr(policy, "observe_transition", None))
        ends_episodes = callable(getattr(policy, "end_episode", None))
        if observes_transitions != ends_episodes:
            raise ValueError("Transition-aware policies must implement both observe_transition and end_episode.")

        self._rng = random.Random(random_seed)

        if policy is not None:
            self._policy = policy
        else:
            assert actions is not None
            self._policy = EpsilonGreedyPolicy(
                actions=actions,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                gamma=gamma,
                state_key=state_key,
                rng=self._rng,
            )

        self._environment_reset = environment_reset
        self._environment_step = environment_step
        self._max_episodes = max_episodes
        self._max_steps_per_episode = max_steps_per_episode
        self._gamma = gamma
        self._value_mode = "custom" if policy is not None else ("state_action" if state_key else "global_action")
        self._uses_transition_updates = observes_transitions
        self._convergence_threshold = convergence_threshold
        self._convergence_episodes = convergence_episodes
        self._trace_detail = trace_detail
        self._reward_stability_criterion = (
            _RewardStabilityCriterion(
                threshold=convergence_threshold,
                stable_episodes=convergence_episodes,
            )
            if convergence_threshold is not None
            else None
        )
        self._random_seed = random_seed
        self._tracer = tracer
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute the reinforcement learning pattern."""
        return self.compile(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def evaluate(
        self,
        *,
        episodes: int = 10,
        max_steps_per_episode: int | None = None,
        environment_reset: EnvironmentResetDelegate | None = None,
        environment_step: EnvironmentStepDelegate | None = None,
        action_selector: EvaluationActionDelegate | None = None,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Evaluate the current policy without updating it.

        The built-in epsilon-greedy policy uses deterministic greedy actions.
        Custom policies can implement ``select_evaluation_action(state)`` or
        callers can pass an explicit ``action_selector``. Independent replicate
        scheduling and statistical analysis intentionally remain responsibilities
        of ``design-research-experiments`` and ``design-research-analysis``.

        Args:
            episodes: Number of evaluation episodes.
            max_steps_per_episode: Optional evaluation step cap. Defaults to the
                training cap configured on the pattern.
            environment_reset: Optional evaluation-only reset callable.
            environment_step: Optional evaluation-only step callable.
            action_selector: Optional deterministic evaluation action callable.
            request_id: Optional evaluation request identifier.
            dependencies: Optional metadata dependencies for the result contract.

        Returns:
            Canonical execution result containing rewards, lengths, and traces.

        Raises:
            ValueError: If evaluation bounds are invalid or a custom policy has no
                explicit evaluation action path.
        """
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
            raise ValueError("episodes must be >= 1.")
        step_limit = self._max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode
        if isinstance(step_limit, bool) or not isinstance(step_limit, int) or step_limit < 1:
            raise ValueError("max_steps_per_episode must be >= 1.")

        resolved_reset = environment_reset or self._environment_reset
        resolved_step = environment_step or self._environment_step
        resolved_action_selector = action_selector or self._resolve_evaluation_action_selector()
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        episode_traces: list[dict[str, object]] = []
        terminated_episodes = 0
        truncated_episodes = 0

        for episode_index in range(episodes):
            reward_total, transitions = self._collect_episode(
                environment_reset=resolved_reset,
                environment_step=resolved_step,
                action_selector=resolved_action_selector,
                max_steps=step_limit,
            )
            final_transition = transitions[-1]
            steps = len(transitions)
            episode_rewards.append(reward_total)
            episode_lengths.append(steps)
            terminated_episodes += int(final_transition.terminated)
            truncated_episodes += int(final_transition.truncated)
            episode_traces.append(
                self._build_episode_trace(
                    episode_index=episode_index,
                    episode_reward=reward_total,
                    transitions=transitions,
                )
            )

        resolved_dependencies = dict(dependencies or {})
        return build_pattern_execution_result(
            success=True,
            final_output={
                "episodes_completed": episodes,
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "mean_reward": sum(episode_rewards) / episodes,
                "mean_steps": sum(episode_lengths) / episodes,
                "min_reward": min(episode_rewards),
                "max_reward": max(episode_rewards),
            },
            terminated_reason="evaluation_completed",
            details={
                "evaluation": True,
                "trace_detail": self._trace_detail,
                "terminated_episodes": terminated_episodes,
                "truncated_episodes": truncated_episodes,
                "episode_traces": episode_traces,
            },
            workflow_payload={},
            artifacts=[],
            request_id=request_id or "rl-evaluation",
            dependencies=resolved_dependencies,
            mode=MODE_REINFORCEMENT_LEARNING,
            metadata={
                "evaluation": True,
                "episodes": episodes,
                "max_steps_per_episode": step_limit,
                "value_mode": self._value_mode,
                "trace_detail": self._trace_detail,
            },
            requested_mode=MODE_REINFORCEMENT_LEARNING,
            resolved_mode=MODE_REINFORCEMENT_LEARNING,
        )

    def compile(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile one reinforcement learning workflow."""
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
            pattern_name="ReinforcementLearningPattern",
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            input_payload={
                **run_context.normalized_input,
                "mode": MODE_REINFORCEMENT_LEARNING,
                "max_episodes": self._max_episodes,
                "max_steps_per_episode": self._max_steps_per_episode,
                "gamma": self._gamma,
                "value_mode": self._value_mode,
                "convergence_threshold": self._convergence_threshold,
                "convergence_episodes": self._convergence_episodes,
                "trace_detail": self._trace_detail,
            },
            workflow_request_id=f"{run_context.request_id}:rl_workflow",
            finalize=lambda workflow_result: _build_reinforcement_learning_result(
                workflow_result=workflow_result,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
                max_episodes=self._max_episodes,
                max_steps_per_episode=self._max_steps_per_episode,
                gamma=self._gamma,
                value_mode=self._value_mode,
                convergence_threshold=self._convergence_threshold,
                convergence_episodes=self._convergence_episodes,
                trace_detail=self._trace_detail,
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
        """Build the workflow for one reinforcement learning training run."""
        _ = prompt, request_id, dependencies
        wrapped_handler = wrap_iteration_handler(
            handler=self._run_iteration,
            error_prefix="ReinforcementLearningPattern episode",
        )
        loop_callbacks = build_loop_callbacks(
            iteration_step_id="rl_episode",
            iteration_handler=wrapped_handler,
        )

        workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
            steps=[
                LoopStep(
                    step_id="rl_loop",
                    steps=(LogicStep(step_id="rl_episode", handler=loop_callbacks.iteration_handler),),
                    max_iterations=self._max_episodes,
                    initial_state=self._get_initial_loop_state(),
                    continue_predicate=loop_callbacks.continue_predicate,
                    state_reducer=loop_callbacks.state_reducer,
                    execution_mode="sequential",
                    failure_policy="propagate_failed_state",
                    retain_iteration_results=False,
                )
            ],
        )
        self.workflow = workflow
        return workflow

    def _get_initial_loop_state(self) -> dict[str, object]:
        """Return the initial state for one bounded training workflow."""
        initial_policy_params = _trace_snapshot(self._policy.get_params())
        return {
            "episode": 0,
            "should_continue": True,
            "episode_rewards": [],
            "best_episode_reward": float("-inf"),
            "best_episode_index": -1,
            "convergence_counter": 0,
            "terminated_reason": None,
            "initial_policy_params": initial_policy_params,
            "final_policy_params": initial_policy_params,
            "episode_traces": [],
        }

    def _run_iteration(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Run one complete episode and reduce it into the loop state."""
        raw_loop_state = context.get("loop_state")
        loop_state = dict(raw_loop_state) if isinstance(raw_loop_state, Mapping) else {}
        episode_index = int(loop_state.get("episode", 0))
        episode_reward, episode_trace = self._run_episode(episode_index)
        episode_rewards = [*list(loop_state.get("episode_rewards", [])), episode_reward]

        best_reward = float(loop_state.get("best_episode_reward", float("-inf")))
        best_index = int(loop_state.get("best_episode_index", -1))
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_index = episode_index

        stability_count = int(loop_state.get("convergence_counter", 0))
        reward_stable = False
        if self._reward_stability_criterion is not None:
            stability_count, reward_stable = self._reward_stability_criterion.observe(
                episode_rewards,
                stability_count,
            )

        max_episodes_reached = (episode_index + 1) >= self._max_episodes
        should_continue = not reward_stable and not max_episodes_reached
        terminated_reason = None
        if reward_stable:
            terminated_reason = "reward_stable"
        elif max_episodes_reached:
            terminated_reason = "max_episodes_reached"

        final_policy_params = loop_state.get("final_policy_params", {})
        if self._trace_detail == "full" or not should_continue:
            final_policy_params = _trace_snapshot(self._policy.get_params())
        if self._trace_detail == "full":
            episode_trace["policy_params"] = final_policy_params

        return {
            "episode": episode_index + 1,
            "should_continue": should_continue,
            "episode_rewards": episode_rewards,
            "best_episode_reward": best_reward,
            "best_episode_index": best_index,
            "convergence_counter": stability_count,
            "terminated_reason": terminated_reason,
            "initial_policy_params": loop_state.get("initial_policy_params", {}),
            "final_policy_params": final_policy_params,
            "episode_traces": [*list(loop_state.get("episode_traces", [])), episode_trace],
        }

    def _run_episode(self, episode_index: int) -> tuple[float, dict[str, object]]:
        """Collect one trajectory, update the policy, and build its trace snapshot."""
        observer = cast(TransitionRLPolicy, self._policy).observe_transition if self._uses_transition_updates else None
        episode_reward, transitions = self._collect_episode(
            environment_reset=self._environment_reset,
            environment_step=self._environment_step,
            action_selector=self._policy.select_action,
            max_steps=self._max_steps_per_episode,
            transition_observer=observer,
        )

        if self._uses_transition_updates:
            raw_update_stats = cast(TransitionRLPolicy, self._policy).end_episode(tuple(transitions))
        else:
            trajectory = [transition.as_legacy_tuple() for transition in transitions]
            raw_update_stats = cast(RLPolicy, self._policy).update(trajectory)
        return episode_reward, self._build_episode_trace(
            episode_index=episode_index,
            episode_reward=episode_reward,
            transitions=transitions,
            update_stats=raw_update_stats,
        )

    def _collect_episode(
        self,
        *,
        environment_reset: EnvironmentResetDelegate,
        environment_step: EnvironmentStepDelegate,
        action_selector: EvaluationActionDelegate,
        max_steps: int,
        transition_observer: Callable[[RLTransition], None] | None = None,
    ) -> tuple[float, list[RLTransition]]:
        """Collect one bounded episode through shared train/evaluate semantics."""
        state = _normalize_state(environment_reset(), name="environment_reset result")
        episode_reward = 0.0
        transitions: list[RLTransition] = []

        for step_number in range(max_steps):
            recorded_state = self._copy_state_for_trajectory(state)
            action = action_selector(state)
            transition = self._step_environment_with(
                environment_step=environment_step,
                state=state,
                action=action,
                recorded_state=recorded_state,
                recorded_action=self._copy_action_for_trajectory(action),
                force_truncated=(step_number + 1 == max_steps),
            )
            transitions.append(transition)
            if transition_observer is not None:
                transition_observer(transition)
            episode_reward += transition.reward
            state = dict(transition.next_state)
            if transition.done:
                break

        return episode_reward, transitions

    def _build_episode_trace(
        self,
        *,
        episode_index: int,
        episode_reward: float,
        transitions: Sequence[RLTransition],
        update_stats: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Build one episode trace at the configured detail level."""
        trace: dict[str, object] = {
            "episode": episode_index,
            "episode_reward": episode_reward,
            "steps": len(transitions),
        }
        if update_stats is not None:
            trace["update_stats"] = _trace_snapshot(update_stats)
        if self._trace_detail != "summary":
            trace["step_traces"] = [
                self._build_transition_trace(step_number, transition)
                for step_number, transition in enumerate(transitions)
            ]
        return trace

    @staticmethod
    def _copy_state_for_trajectory(state: RLState) -> dict[str, object]:
        """Return a defensive state copy for the policy trajectory."""
        copied_state = _copy_experience_value(state, name="state")
        if not isinstance(copied_state, Mapping):
            raise TypeError("copied state must remain a mapping.")
        return dict(copied_state)

    @staticmethod
    def _copy_action_for_trajectory(action: object) -> RLAction:
        """Validate and defensively copy one selected action."""
        if not isinstance(action, (str, Mapping)):
            raise TypeError("policy.select_action must return a string or mapping.")
        copied_action = _copy_experience_value(action, name="action")
        if not isinstance(copied_action, (str, Mapping)):
            raise TypeError("copied action must remain a string or mapping.")
        return dict(copied_action) if isinstance(copied_action, Mapping) else copied_action

    def _resolve_evaluation_action_selector(self) -> EvaluationActionDelegate:
        """Return the policy's explicit deterministic evaluation selector."""
        selector = getattr(self._policy, "select_evaluation_action", None)
        if not callable(selector):
            raise ValueError("Custom policy evaluation requires action_selector or select_evaluation_action(state).")
        return cast(EvaluationActionDelegate, selector)

    @staticmethod
    def _build_transition_trace(step_number: int, transition: RLTransition) -> dict[str, object]:
        """Return one bounded, compatibility-preserving transition trace."""
        return {
            "step_num": step_number,
            "state": _trace_snapshot(transition.state),
            "action": _trace_snapshot(transition.action),
            "reward": transition.reward,
            "next_state": _trace_snapshot(transition.next_state),
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "done": transition.done,
            "info": _trace_snapshot(transition.info),
        }

    def _step_environment(self, state: RLState, action: RLAction) -> RLTransition:
        """Execute one configured environment step and return a rich transition."""
        return self._step_environment_with(
            environment_step=self._environment_step,
            state=state,
            action=action,
            recorded_state=self._copy_state_for_trajectory(state),
            recorded_action=self._copy_action_for_trajectory(action),
            force_truncated=False,
        )

    @classmethod
    def _step_environment_with(
        cls,
        *,
        environment_step: EnvironmentStepDelegate,
        state: RLState,
        action: RLAction,
        recorded_state: RLState,
        recorded_action: RLAction,
        force_truncated: bool,
    ) -> RLTransition:
        """Normalize legacy or Gymnasium-style environment output."""
        raw_result = environment_step(state, action)
        if not isinstance(raw_result, tuple):
            raise TypeError("environment_step must return a tuple with 3 or 5 values.")
        if len(raw_result) == 3:
            next_state, reward, done = raw_result
            terminated = done
            truncated = False
            info: object = {}
            termination_name = "done"
        elif len(raw_result) == 5:
            next_state, reward, terminated, truncated, info = raw_result
            termination_name = "terminated"
        else:
            raise ValueError("environment_step must return exactly 3 or 5 values.")

        normalized_state = _normalize_state(next_state, name="environment_step next_state")
        try:
            normalized_reward = float(reward)
        except (TypeError, ValueError) as exc:
            raise TypeError("environment_step reward must be numeric.") from exc
        if not math.isfinite(normalized_reward):
            raise ValueError("environment_step reward must be finite.")
        if not isinstance(terminated, bool):
            raise TypeError(f"environment_step {termination_name} must be a bool.")
        if not isinstance(truncated, bool):
            raise TypeError("environment_step truncated must be a bool.")
        if not isinstance(info, Mapping):
            raise TypeError("environment_step info must be a mapping.")
        normalized_truncated = truncated or (force_truncated and not terminated)
        return RLTransition(
            state=cls._copy_state_for_trajectory(recorded_state),
            action=cls._copy_action_for_trajectory(recorded_action),
            reward=normalized_reward,
            next_state=cls._copy_state_for_trajectory(normalized_state),
            terminated=terminated,
            truncated=normalized_truncated,
            info=dict(info),
        )


def _build_reinforcement_learning_result(
    *,
    workflow_result: ExecutionResult,
    request_id: str,
    dependencies: Mapping[str, object],
    max_episodes: int,
    max_steps_per_episode: int,
    gamma: float,
    value_mode: str,
    convergence_threshold: float | None,
    convergence_episodes: int,
    trace_detail: str,
    random_seed: int | None,
) -> ExecutionResult:
    """Build the final ExecutionResult from one reinforcement learning workflow run."""
    loop_step_result = workflow_result.step_results.get("rl_loop")
    loop_output = dict(loop_step_result.output) if loop_step_result is not None else {}
    final_state_raw = loop_output.get("final_state")
    final_state = dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}
    workflow_output = build_workflow_output_payload(workflow_result)
    workflow_payload = workflow_output["workflow"]
    terminated_reason = str(
        final_state.get("terminated_reason", loop_output.get("terminated_reason"))
        if workflow_result.success
        else "workflow_failure"
    )
    raw_episode_traces = final_state.get("episode_traces", [])
    episode_traces = list(raw_episode_traces) if isinstance(raw_episode_traces, Sequence) else []
    final_policy_params: object = final_state.get(
        "final_policy_params",
        final_state.get("initial_policy_params", {}),
    )
    return build_pattern_execution_result(
        success=workflow_result.success,
        final_output={
            "best_episode_reward": final_state.get("best_episode_reward"),
            "best_episode_index": final_state.get("best_episode_index"),
            "episodes_completed": final_state.get("episode"),
            "final_policy_params": final_policy_params,
            "episode_rewards": final_state.get("episode_rewards"),
        },
        terminated_reason=terminated_reason,
        details={
            "max_episodes": max_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "gamma": gamma,
            "value_mode": value_mode,
            "trace_detail": trace_detail,
            "initial_policy_params": final_state.get("initial_policy_params", {}),
            "episode_traces": episode_traces,
        },
        workflow_payload=workflow_payload if isinstance(workflow_payload, Mapping) else {},
        artifacts=workflow_output["artifacts"],
        request_id=request_id,
        dependencies=dependencies,
        mode=MODE_REINFORCEMENT_LEARNING,
        metadata={
            "max_episodes": max_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "gamma": gamma,
            "value_mode": value_mode,
            "convergence_threshold": convergence_threshold,
            "convergence_episodes": convergence_episodes,
            "trace_detail": trace_detail,
            "random_seed": random_seed,
        },
        requested_mode=MODE_REINFORCEMENT_LEARNING,
        resolved_mode=MODE_REINFORCEMENT_LEARNING,
    )


__all__ = [
    "EnvironmentResetDelegate",
    "EnvironmentStepDelegate",
    "EnvironmentStepResult",
    "EpsilonGreedyPolicy",
    "EvaluationActionDelegate",
    "RLAction",
    "RLPolicy",
    "RLState",
    "RLTransition",
    "ReinforcementLearningPattern",
    "Trajectory",
    "TransitionRLPolicy",
]
