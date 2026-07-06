"""Tests for the reinforcement learning pattern."""

from __future__ import annotations

import pytest

from design_research_agents._implementations import (
    EpsilonGreedyPolicy,
    ReinforcementLearningPattern,
)
from design_research_agents._implementations._patterns._reinforcement_learning_pattern import (
    EnvironmentResetDelegate,
    EnvironmentStepDelegate,
    RLState,
    Trajectory,
)

# ---------------------- Test environments ----------------------


def _bandit_env() -> tuple[EnvironmentResetDelegate, EnvironmentStepDelegate]:
    """Single-step, state-independent bandit: ``good`` earns +1, else -1.

    The optimal action is fixed and independent of state, so the default
    state-independet policy provably converges to it."""

    def reset() -> dict[str, object]:
        return {"t": 0}

    def step(state: RLState, action: str) -> tuple[dict[str, object], float, bool]:
        reward = 1.0 if action == "good" else -1.0
        return {"t": 1}, reward, True

    return reset, step


def _grid_env() -> tuple[EnvironmentResetDelegate, EnvironmentStepDelegate]:
    """Line-walk grid: start at 0, reach the goal at 3 by moving ``right``."""

    def reset() -> dict[str, object]:
        return {"position": 0}

    def step(state: RLState, action: str) -> tuple[dict[str, object], float, bool]:
        pos = int(state["position"])
        if action == "right":
            pos += 1
        elif action == "left":
            pos -= 1
        reward = 10.0 if pos == 3 else -1.0
        return {"position": pos}, reward, pos == 3

    return reset, step


# ---------------------- Test environments ----------------------


def test_pattern_validates_max_episodes() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="max_episodes"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            actions=["good", "bad"],
            max_episodes=0,
        )


def test_pattern_validates_max_steps_per_episode() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="max_steps_per_episode"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            actions=["good", "bad"],
            max_steps_per_episode=0,
        )


def test_pattern_validates_gamma() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="gamma"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            actions=["good", "bad"],
            gamma=0.0,
        )


def test_pattern_validates_convergence_threshold() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="convergence_threshold"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            actions=["good", "bad"],
            convergence_threshold=-0.1,
        )


def test_pattern_validates_convergence_episodes() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="convergence_episodes"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            actions=["good", "bad"],
            convergence_episodes=0,
        )


def test_pattern_requires_policy_or_actions() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="Either policy or actions"):
        ReinforcementLearningPattern(environment_reset=reset, environment_step=step)


def test_pattern_builds_default_policy_from_actions() -> None:
    reset, step = _bandit_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["good", "bad"],
    )
    assert isinstance(pattern._policy, EpsilonGreedyPolicy)


# ---------------------- EpsilonGreedyPolicy unit tests ----------------------


def test_policy_validates_non_empty_actions() -> None:
    with pytest.raises(ValueError, match="actions"):
        EpsilonGreedyPolicy(actions=[])


def test_policy_validates_epsilon_range() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyPolicy(actions=["a", "b"], epsilon=-0.1)
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyPolicy(actions=["a", "b"], epsilon=1.1)


def test_policy_greedy_selection_picks_highest_q_value() -> None:
    policy = EpsilonGreedyPolicy(actions=["a", "b"], epsilon=0.0)
    policy.update([({"s": 0}, "a", 0.0), ({"s": 0}, "b", 5.0)])
    # With epsilon=0, policy is deterministic and must exploit the best action.
    assert policy.select_action({"s": 0}) == "b"


def test_policy_update_on_empty_trajectory_is_noop() -> None:
    policy = EpsilonGreedyPolicy(actions=["a"])
    assert policy.update([]) == {"mean_return": 0.0}


def test_policy_update_computes_discounted_returns() -> None:
    policy = EpsilonGreedyPolicy(actions=["a", "b"], epsilon=1.0, epsilon_decay=0.5, gamma=0.5)
    trajectory: Trajectory = [({"s": 0}, "a", 1.0), ({"s": 0}, "b", 2.0)]
    stats = policy.update(trajectory)
    params = policy.get_params()
    # return[a] = 1.0 + 0.5 * 2.0 = 2.0
    # return[b] = 2.0
    assert params["q_values"]["a"] == pytest.approx(2.0)
    assert params["q_values"]["b"] == pytest.approx(2.0)
    assert stats["mean_return"] == pytest.approx(2.0)
    # epsilon decays after an update
    assert params["epsilon"] == pytest.approx(0.5)


# ---------------------- Convergence and learning ----------------------


def test_policy_converges_on_bandit_and_learns_from_reward() -> None:
    reset, step = _bandit_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["good", "bad"],
        max_episodes=200,
        max_steps_per_episode=1,
        gamma=1.0,
        # Pure greedy (epsilon=0): once "good" is sampled, its Q-value strictly
        # dominates "bad" and the policy will always select it thereafter,
        # resulting in convergence.
        epsilon=0.0,
        convergence_threshold=1e-9,
        convergence_episodes=5,
        random_seed=0,
    )
    result = pattern.run("Pick the good arm.", request_id="rl-test-bandit")

    assert result.success
    assert result.output["terminated_reason"] == "converged"

    final_output = result.output["final_output"]
    q_values = final_output["final_policy_params"]["q_values"]
    # Single-step return for "good" is always +1.0, so its MC estimate is 1.0.
    assert q_values["good"] == pytest.approx(1.0)
    # "bad" is either never sampled (0.0) or estimated at its true value (-1.0).
    assert q_values["bad"] <= 0.0


def test_final_policy_params_reflect_learned_not_initial_params() -> None:
    """Regression: policy_params_history must record post-update params each episode."""
    reset, step = _bandit_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["good", "bad"],
        max_episodes=25,
        max_steps_per_episode=1,
        epsilon=1.0,
        epsilon_decay=0.5,
        epsilon_min=0.0,
        random_seed=0,
    )
    result = pattern.run("learn", request_id="rl-test-history")
    details = result.output["details"]
    history = details["policy_params_history"]
    episodes_completed = result.output["final_output"]["episodes_completed"]

    # Seeded with initial snapshot, then one snapshot appended per episode.
    assert len(history) == episodes_completed + 1
    # Initial snapshot is untrained; later snapshots reflect learning.
    assert all(value == 0.0 for value in history[0]["q_values"].values())
    assert any(value != 0.0 for value in history[-1]["q_values"].values())
    # final_policy_params is surfaced from the learned tail, not the initial snapshot
    # (asserted independently of history[-1] rather than by definitional equality).
    final_q_values = result.output["final_output"]["final_policy_params"]["q_values"]
    assert any(value != 0.0 for value in final_q_values.values())
    # Epsilon anneals monotonically across run (1.0 initial -> decayed).
    epsilons = [snapshot["epsilon"] for snapshot in history]
    assert epsilons == sorted(epsilons, reverse=True)
    assert epsilons[0] == pytest.approx(1.0)


def test_reaches_max_episodes_when_never_converging() -> None:
    reset, step = _grid_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["left", "right", "stay"],
        max_episodes=3,
        max_steps_per_episode=10,
        # Threshold of exactly 0 can never be satisfied.
        convergence_threshold=0.0,
        random_seed=1,
    )
    result = pattern.run("navigate", request_id="rl-test-max")

    assert result.success
    assert result.output["terminated_reason"] == "max_episodes_reached"
    assert result.output["final_output"]["episodes_completed"] == 3


def test_episode_stops_at_max_steps_when_never_done() -> None:
    def reset() -> dict[str, object]:
        return {"t": 0}

    def step(state: RLState, action: str) -> tuple[dict[str, object], float, bool]:
        return {"t": 1}, -1.0, False  # never terminates on its own

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["a"],
        max_episodes=1,
        max_steps_per_episode=4,
        convergence_threshold=0.0,
        random_seed=0,
    )

    result = pattern.run("cap", request_id="rl-test-cap")
    trace = result.output["details"]["episode_traces"][0]

    assert trace["steps"] == 4
    assert len(trace["step_traces"]) == 4


# ---------------------- Trace capture ----------------------


def test_traces_capture_states_actions_rewards_and_updates() -> None:
    reset, step = _grid_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["left", "right", "stay"],
        max_episodes=4,
        max_steps_per_episode=10,
        convergence_threshold=0.0,
        random_seed=7,
    )
    result = pattern.run("navigate", request_id="rl-test-trace")
    traces = result.output["details"]["episode_traces"]

    assert len(traces) == result.output["final_output"]["episodes_completed"]
    first = traces[0]
    assert set(first) >= {"episode", "episode_reward", "steps", "step_traces", "update_stats", "policy_params"}
    assert first["steps"] >= 1

    first_step = first["step_traces"][0]
    assert set(first_step) >= {"step_num", "state", "action", "reward", "next_state", "done"}
    assert "position" in first_step["state"]
    assert first_step["action"] in {"left", "right", "stay"}
    assert isinstance(first_step["reward"], float)
    assert "mean_return" in first["update_stats"]


def test_traces_are_isolated_from_in_place_mutating_environment() -> None:
    def reset() -> dict[str, object]:
        return {"position": 0}

    def step(state: dict[str, object], action: str) -> tuple[dict[str, object], float, bool]:
        # Naive env: mutates its argument in place and returns the same object.
        state["position"] = int(state["position"]) + 1
        pos = int(state["position"])
        return state, (10.0 if pos == 3 else -1.0), pos == 3

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["step"],
        max_episodes=1,
        max_steps_per_episode=5,
        convergence_threshold=0.0,
        random_seed=0,
    )

    result = pattern.run("walk", request_id="rl-test-alias")
    steps_trace = result.output["details"]["episode_traces"][0]["step_traces"]

    # Each recorded pre-transtion state must reflect the position at that step,
    # not the mutated end-of-episode value that in-place mutation would leak.
    positions = [int(trace["state"]["position"]) for trace in steps_trace]
    assert positions == [0, 1, 2]


# ---------------------- Custom policy plug-in ----------------------


def test_custom_policy_is_used_and_snapshotted_each_episode() -> None:
    class _RecordingPolicy:
        def __init__(self) -> None:
            self.updates = 0

        def select_action(self, state: RLState) -> str:
            return "noop"

        def update(self, trajectory: Trajectory) -> dict[str, object]:
            self.updates += 1
            return {"mean_return": 0.0}

        def get_params(self) -> dict[str, object]:
            return {"updates": self.updates}

    def reset() -> dict[str, object]:
        return {"t": 0}

    def step(state: RLState, action: str) -> tuple[dict[str, object], float, bool]:
        return {"t": 1}, 0.0, True

    policy = _RecordingPolicy()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        policy=policy,
        max_episodes=3,
        max_steps_per_episode=1,
    )

    assert pattern._policy is policy
    result = pattern.run("run", request_id="rl-test-custom")

    assert policy.updates == 3
    assert result.output["final_output"]["final_policy_params"] == {"updates": 3}


def test_custom_policy_supports_mapping_actions() -> None:
    class _ParamPolicy:
        def select_action(self, state: RLState) -> str:
            return {"force": 1.0}

        def update(self, trajectory: Trajectory) -> dict[str, object]:
            return {"mean_return": 0.0}

        def get_params(self) -> dict[str, object]:
            return {}

    def reset() -> dict[str, object]:
        return {"t": 0}

    def step(state: RLState, action: str) -> tuple[dict[str, object], float, bool]:
        return {"t": 1}, 0.0, True

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        policy=_ParamPolicy(),
        max_episodes=1,
        max_steps_per_episode=1,
    )

    result = pattern.run("params", request_id="rl-test-mapping")
    step_traces = result.output["details"]["episode_traces"][0]["step_traces"]

    # RLAction = str | Mapping: a dict-valued action must round-trip through the trace.
    assert step_traces[0]["action"] == {"force": 1.0}


# ---------------------- Failure path ----------------------


def test_workflow_failure_when_environment_raises() -> None:
    def reset() -> dict[str, object]:
        raise {"t": 0}

    def step(state: RLState, action: str) -> tuple[dict[str, object], float, bool]:
        return RuntimeError("environment boom")

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["a"],
        max_episodes=3,
        max_steps_per_episode=1,
    )

    result = pattern.run("boom", request_id="rl-test-failure")

    assert not result.success
    assert result.output["terminated_reason"] == "workflow_failure"
