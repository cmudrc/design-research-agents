"""Reusable ``reinforcement learning`` orchestration chunk."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

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
# One episode's experience of (state, action, reward) tuples
Trajectory = list[tuple[RLState, RLAction, float]]

# Returns initial state for new episode
EnvironmentResetDelegate = Callable[[], RLState]
# (state, action) -> (next_state, reward, done)
EnvironmentStepDelegate = Callable[[RLState, RLAction], tuple[RLState, float, bool]]


class RLPolicy(Protocol):
    """Protocol for pluggable RL policies."""

    def select_action(self, state: RLState) -> RLAction:
        """Select an action for the current state."""
        ...

    def update(self, trajectory: Trajectory) -> dict[str, object]:
        """Update the policy from one episode and return JSON-safe statistics."""
        ...

    def get_params(self) -> dict[str, object]:
        """Return a fresh, JSON-safe snapshot of the policy parameters."""
        ...


class EpsilonGreedyPolicy:
    """Simple epsilon-greedy policy with Monte Carlo Q-value action-value updates.

    Note: this policy is state-independent. It keeps one Q-value per action and
    ignores the state. This is suitable for simple environments with a small
    number of discrete actions; plug in custom ``RLPolicy`` otherwise.
    """

    def __init__(
        self,
        *,
        actions: Sequence[str],
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.0,
        gamma: float = 0.99,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize a state-independent action-value policy.

        Args:
            actions: Unique, non-empty discrete action names.
            epsilon: Initial probability of selecting a random action.
            epsilon_decay: Multiplicative epsilon decay after each episode.
            epsilon_min: Lower bound applied when epsilon decays.
            gamma: Discount factor used for Monte Carlo returns.
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
        self._rng = rng or random.Random()
        self._q_values: dict[str, float] = {action: 0.0 for action in normalized_actions}
        self._action_counts: dict[str, int] = {action: 0 for action in normalized_actions}

    @property
    def epsilon(self) -> float:
        """Return the current exploration probability."""
        return self._epsilon

    def select_action(self, state: RLState) -> str:
        """Select random action with epsilon probability, else action with highest Q-value."""
        _ = state
        if self._rng.random() < self._epsilon:
            return self._rng.choice(self._actions)
        best_value = max(self._q_values.values())
        best_actions = [a for a, v in self._q_values.items() if v == best_value]
        return self._rng.choice(best_actions)

    def update(self, trajectory: Trajectory) -> dict[str, object]:
        """Compute discounted returns from episode and update Q-values."""
        if not trajectory:
            return {"mean_return": 0.0}

        n = len(trajectory)
        returns = [0.0] * n
        returns[-1] = trajectory[-1][2]
        for t in range(n - 2, -1, -1):
            returns[t] = trajectory[t][2] + self._gamma * returns[t + 1]

        for t in range(n):
            _, action, _ = trajectory[t]
            if not isinstance(action, str) or action not in self._q_values:
                raise ValueError("trajectory contains an action not configured for this policy.")
            self._action_counts[action] += 1
            count = self._action_counts[action]
            self._q_values[action] += (returns[t] - self._q_values[action]) / count

        self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)

        return {
            "mean_return": sum(returns) / len(returns),
            "epsilon": self._epsilon,
        }

    def get_params(self) -> dict[str, object]:
        """Return a fresh snapshot of learned action values and exploration state."""
        return {
            "q_values": dict(self._q_values),
            "action_counts": dict(self._action_counts),
            "epsilon": self._epsilon,
        }


class ReinforcementLearningPattern(Delegate):
    """Reinforcement learning pattern with episodic agent-environment loop."""

    def __init__(
        self,
        *,
        environment_reset: EnvironmentResetDelegate,
        environment_step: EnvironmentStepDelegate,
        policy: RLPolicy | None = None,
        actions: Sequence[str] | None = None,
        max_episodes: int = 100,
        max_steps_per_episode: int = 50,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.0,
        convergence_threshold: float = 0.01,
        convergence_episodes: int = 5,
        random_seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store the environment, policy, and bounded training configuration.

        Args:
            environment_reset: Callable returning the initial state for each episode.
            environment_step: Callable applying one action and returning next state,
                reward, and an episode-complete flag.
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
            convergence_threshold: Maximum reward change counted as stable.
            convergence_episodes: Consecutive stable changes required to stop early.
            random_seed: Optional seed for default-policy action selection.
            tracer: Optional workflow tracer.

        Raises:
            ValueError: If the training bounds or policy selection are invalid.
        """
        if max_episodes < 1:
            raise ValueError("max_episodes must be >= 1.")
        if max_steps_per_episode < 1:
            raise ValueError("max_steps_per_episode must be >= 1.")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1].")
        if convergence_threshold < 0:
            raise ValueError("convergence_threshold must be >= 0.")
        if convergence_episodes < 1:
            raise ValueError("convergence_episodes must be >= 1.")
        if policy is None and not actions:
            raise ValueError("Either policy or actions must be provided.")
        if policy is not None and actions is not None:
            raise ValueError("policy and actions are mutually exclusive.")

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
                rng=self._rng,
            )

        self._environment_reset = environment_reset
        self._environment_step = environment_step
        self._max_episodes = max_episodes
        self._max_steps_per_episode = max_steps_per_episode
        self._gamma = gamma
        self._convergence_threshold = convergence_threshold
        self._convergence_episodes = convergence_episodes
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
                "convergence_threshold": self._convergence_threshold,
                "convergence_episodes": self._convergence_episodes,
            },
            workflow_request_id=f"{run_context.request_id}:rl_workflow",
            finalize=lambda workflow_result: _build_reinforcement_learning_result(
                workflow_result=workflow_result,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
                max_episodes=self._max_episodes,
                max_steps_per_episode=self._max_steps_per_episode,
                gamma=self._gamma,
                convergence_threshold=self._convergence_threshold,
                convergence_episodes=self._convergence_episodes,
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

        def _get_initial_loop_state() -> dict[str, object]:
            return {
                "episode": 0,
                "should_continue": True,
                "episode_rewards": [],
                "best_episode_reward": float("-inf"),
                "best_episode_index": -1,
                "convergence_counter": 0,
                "terminated_reason": None,
                "policy_params_history": [self._policy.get_params()],
                "episode_traces": [],
            }

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
            raw_loop_state = context.get("loop_state")
            loop_state = dict(raw_loop_state) if isinstance(raw_loop_state, Mapping) else {}
            episode_idx = int(loop_state.get("episode", 0))

            state = dict(self._environment_reset())
            trajectory: list[tuple[RLState, RLAction, float]] = []
            episode_reward = 0.0
            step_traces: list[dict[str, object]] = []

            for step_num in range(self._max_steps_per_episode):
                recorded_state = dict(state)

                action = self._policy.select_action(state)
                next_state, reward, done = self._environment_step(state, action)
                next_state = dict(next_state)
                recorded_action = dict(action) if isinstance(action, Mapping) else action
                normalized_reward = float(reward)

                trajectory.append((recorded_state, recorded_action, normalized_reward))
                step_traces.append(
                    {
                        "step_num": step_num,
                        "state": recorded_state,
                        "action": recorded_action,
                        "reward": normalized_reward,
                        "next_state": next_state,
                        "done": done,
                    }
                )
                episode_reward += normalized_reward
                state = next_state

                if done:
                    break

            update_stats = self._policy.update(trajectory)

            episode_rewards = [*list(loop_state.get("episode_rewards", [])), episode_reward]
            best_reward = float(loop_state.get("best_episode_reward", float("-inf")))
            best_idx = int(loop_state.get("best_episode_index", -1))
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_idx = episode_idx

            convergence_counter = int(loop_state.get("convergence_counter", 0))
            terminated_reason = None
            should_continue = True

            if len(episode_rewards) >= 2:
                if abs(episode_rewards[-1] - episode_rewards[-2]) < self._convergence_threshold:
                    convergence_counter += 1
                else:
                    convergence_counter = 0

            if convergence_counter >= self._convergence_episodes:
                should_continue = False
                terminated_reason = "converged"

            elif (episode_idx + 1) >= self._max_episodes:
                should_continue = False
                terminated_reason = "max_episodes_reached"

            episode_trace = {
                "episode": episode_idx,
                "episode_reward": episode_reward,
                "steps": len(trajectory),
                "step_traces": step_traces,
                "update_stats": update_stats,
                "policy_params": self._policy.get_params(),
            }

            return {
                "episode": episode_idx + 1,
                "should_continue": should_continue,
                "episode_rewards": episode_rewards,
                "best_episode_reward": best_reward,
                "best_episode_index": best_idx,
                "convergence_counter": convergence_counter,
                "terminated_reason": terminated_reason,
                "policy_params_history": [
                    *list(loop_state.get("policy_params_history", [])),
                    self._policy.get_params(),
                ],
                "episode_traces": [*list(loop_state.get("episode_traces", [])), episode_trace],
            }

        wrapped_handler = wrap_iteration_handler(
            handler=_run_iteration,
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


def _build_reinforcement_learning_result(
    *,
    workflow_result: ExecutionResult,
    request_id: str,
    dependencies: Mapping[str, object],
    max_episodes: int,
    max_steps_per_episode: int,
    gamma: float,
    convergence_threshold: float,
    convergence_episodes: int,
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
    policy_params_history = final_state.get("policy_params_history", [])
    return build_pattern_execution_result(
        success=workflow_result.success,
        final_output={
            "best_episode_reward": final_state.get("best_episode_reward"),
            "best_episode_index": final_state.get("best_episode_index"),
            "episodes_completed": final_state.get("episode"),
            "final_policy_params": policy_params_history[-1] if policy_params_history else {},
            "episode_rewards": final_state.get("episode_rewards"),
        },
        terminated_reason=terminated_reason,
        details={
            "max_episodes": max_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "gamma": gamma,
            "episode_traces": final_state.get("episode_traces"),
            "policy_params_history": policy_params_history,
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
            "convergence_threshold": convergence_threshold,
            "convergence_episodes": convergence_episodes,
            "random_seed": random_seed,
        },
        requested_mode=MODE_REINFORCEMENT_LEARNING,
        resolved_mode=MODE_REINFORCEMENT_LEARNING,
    )


__all__ = [
    "EnvironmentResetDelegate",
    "EnvironmentStepDelegate",
    "EpsilonGreedyPolicy",
    "RLAction",
    "RLPolicy",
    "RLState",
    "ReinforcementLearningPattern",
    "Trajectory",
]
