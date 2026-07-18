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
    RLAction,
    RLState,
    Trajectory,
    _normalize_state,
    _RewardStabilityCriterion,
    _trace_snapshot,
)

# ---------------------- Validation ----------------------


def _bandit_env() -> tuple[EnvironmentResetDelegate, EnvironmentStepDelegate]:
    """Single-step, state-independent bandit: ``good`` earns +1, else -1.

    The optimal action is fixed and independent of state, so the default
    state-independent policy provably converges to it."""

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
    for invalid_value in (0, 1.5, True):
        with pytest.raises(ValueError, match="max_episodes"):
            ReinforcementLearningPattern(
                environment_reset=reset,
                environment_step=step,
                actions=["good", "bad"],
                max_episodes=invalid_value,  # type: ignore[arg-type]
            )


def test_pattern_validates_max_steps_per_episode() -> None:
    reset, step = _bandit_env()
    for invalid_value in (0, 1.5, True):
        with pytest.raises(ValueError, match="max_steps_per_episode"):
            ReinforcementLearningPattern(
                environment_reset=reset,
                environment_step=step,
                actions=["good", "bad"],
                max_steps_per_episode=invalid_value,  # type: ignore[arg-type]
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
    for invalid_value in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="convergence_threshold"):
            ReinforcementLearningPattern(
                environment_reset=reset,
                environment_step=step,
                actions=["good", "bad"],
                convergence_threshold=invalid_value,
            )


def test_pattern_validates_convergence_episodes() -> None:
    reset, step = _bandit_env()
    for invalid_value in (0, 1.5, True):
        with pytest.raises(ValueError, match="convergence_episodes"):
            ReinforcementLearningPattern(
                environment_reset=reset,
                environment_step=step,
                actions=["good", "bad"],
                convergence_episodes=invalid_value,  # type: ignore[arg-type]
            )


def test_pattern_validates_trace_detail() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="trace_detail"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            actions=["good", "bad"],
            trace_detail="verbose",
        )


def test_pattern_requires_policy_or_actions() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="Either policy or actions"):
        ReinforcementLearningPattern(environment_reset=reset, environment_step=step)


def test_pattern_requires_actions_for_state_key_mode() -> None:
    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="state_key requires actions"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            state_key=lambda state: str(state["t"]),
        )


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


@pytest.mark.parametrize("actions", [[""], ["a", "a"]])
def test_policy_validates_action_names(actions: list[str]) -> None:
    with pytest.raises(ValueError, match="actions"):
        EpsilonGreedyPolicy(actions=actions)


def test_policy_validates_epsilon_range() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyPolicy(actions=["a", "b"], epsilon=-0.1)
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyPolicy(actions=["a", "b"], epsilon=1.1)
    with pytest.raises(ValueError, match="epsilon_min"):
        EpsilonGreedyPolicy(actions=["a", "b"], epsilon=0.2, epsilon_min=0.3)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("epsilon_decay", -0.1), ("epsilon_min", -0.1), ("gamma", 1.1)],
)
def test_policy_validates_learning_parameters(keyword: str, value: float) -> None:
    with pytest.raises(ValueError, match=keyword):
        EpsilonGreedyPolicy(actions=["a", "b"], **{keyword: value})


def test_pattern_rejects_policy_and_actions_together() -> None:
    reset, step = _bandit_env()
    policy = EpsilonGreedyPolicy(actions=["good", "bad"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            policy=policy,
            actions=["good", "bad"],
        )


def test_pattern_rejects_state_key_with_custom_policy() -> None:
    reset, step = _bandit_env()
    policy = EpsilonGreedyPolicy(actions=["good", "bad"])
    with pytest.raises(ValueError, match="state_key"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            policy=policy,
            state_key=lambda state: str(state["t"]),
        )


def test_policy_greedy_selection_picks_highest_q_value() -> None:
    policy = EpsilonGreedyPolicy(actions=["a", "b"], epsilon=0.0)
    policy.update([({"s": 0}, "a", 0.0), ({"s": 0}, "b", 5.0)])
    # With epsilon=0, policy is deterministic and must exploit the best action.
    assert policy.select_action({"s": 0}) == "b"


def test_policy_update_on_empty_trajectory_is_noop() -> None:
    policy = EpsilonGreedyPolicy(actions=["a"])
    assert policy.update([]) == {"mean_return": 0.0}


def test_policy_rejects_trajectory_actions_outside_its_action_space() -> None:
    policy = EpsilonGreedyPolicy(actions=["known"])
    with pytest.raises(ValueError, match="not configured"):
        policy.update([({"s": 0}, "unknown", 1.0)])


def test_policy_update_computes_discounted_returns() -> None:
    policy = EpsilonGreedyPolicy(actions=["a", "b"], epsilon=1.0, epsilon_decay=0.5, gamma=0.5)
    trajectory: Trajectory = [({"s": 0}, "a", 1.0), ({"s": 0}, "b", 2.0)]
    stats = policy.update(trajectory)
    params = policy.get_params()
    # return[a] = 1.0 + 0.5 * 2.0 = 2.0
    # return[b] = 2.0
    assert params["value_mode"] == "global_action"
    assert params["action_values"]["a"] == pytest.approx(2.0)
    assert params["action_values"]["b"] == pytest.approx(2.0)
    assert stats["mean_return"] == pytest.approx(2.0)
    assert stats["value_mode"] == "global_action"
    # epsilon decays after an update
    assert params["epsilon"] == pytest.approx(0.5)
    assert policy.epsilon == pytest.approx(0.5)


def test_policy_estimates_tabular_state_action_values() -> None:
    policy = EpsilonGreedyPolicy(
        actions=["explore", "refine"],
        epsilon=0.0,
        gamma=0.5,
        state_key=lambda state: str(state["stage"]),
    )
    trajectory: Trajectory = [
        ({"stage": "concept"}, "explore", 1.0),
        ({"stage": "detail"}, "refine", 2.0),
    ]

    stats = policy.update(trajectory)
    params = policy.get_params()

    assert stats["value_mode"] == "state_action"
    assert params["value_mode"] == "state_action"
    assert params["q_values"]["concept"]["explore"] == pytest.approx(2.0)
    assert params["q_values"]["detail"]["refine"] == pytest.approx(2.0)
    assert params["state_action_counts"]["concept"]["explore"] == 1
    assert policy.select_action({"stage": "concept"}) == "explore"
    assert policy.select_action({"stage": "detail"}) == "refine"


@pytest.mark.parametrize("invalid_key", ["", 7])
def test_policy_validates_state_keys(invalid_key: object) -> None:
    policy = EpsilonGreedyPolicy(
        actions=["a"],
        state_key=lambda state: invalid_key,  # type: ignore[return-value]
    )

    with pytest.raises(ValueError, match="state_key"):
        policy.select_action({"stage": "concept"})


def test_pattern_exposes_tabular_value_mode_in_canonical_result() -> None:
    def reset() -> dict[str, object]:
        return {"stage": "concept"}

    def step(state: RLState, action: str) -> tuple[dict[str, object], float, bool]:
        return {"stage": "complete"}, (1.0 if action == "explore" else 0.0), True

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["explore", "refine"],
        state_key=lambda state: str(state["stage"]),
        epsilon=0.0,
        max_episodes=1,
        random_seed=0,
    )

    result = pattern.run("learn by stage", request_id="rl-test-tabular-pattern")

    assert result.success
    assert result.output["details"]["value_mode"] == "state_action"
    assert result.metadata["value_mode"] == "state_action"
    assert "concept" in result.output["final_output"]["final_policy_params"]["q_values"]


# ---------------------- Convergence and learning ----------------------


def test_reward_stability_is_opt_in_and_does_not_claim_policy_convergence() -> None:
    reset, step = _bandit_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["good", "bad"],
        max_episodes=200,
        max_steps_per_episode=1,
        gamma=1.0,
        # Pure greedy (epsilon=0): once "good" is sampled, its action value strictly
        # dominates "bad" and the policy will always select it thereafter,
        # resulting in a stable reward signal.
        epsilon=0.0,
        convergence_threshold=1e-9,
        convergence_episodes=5,
        random_seed=0,
    )
    result = pattern.run("Pick the good arm.", request_id="rl-test-bandit")

    assert result.success
    assert result.output["terminated_reason"] == "reward_stable"

    final_output = result.output["final_output"]
    action_values = final_output["final_policy_params"]["action_values"]
    # Single-step return for "good" is always +1.0, so its MC estimate is 1.0.
    assert action_values["good"] == pytest.approx(1.0)
    # "bad" is either never sampled (0.0) or estimated at its true value (-1.0).
    assert action_values["bad"] <= 0.0


def test_default_run_does_not_stop_on_stable_rewards() -> None:
    reset, step = _bandit_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["good"],
        max_episodes=7,
        max_steps_per_episode=1,
        epsilon=0.0,
        random_seed=0,
    )

    result = pattern.run("learn", request_id="rl-test-no-default-stability")

    assert result.output["terminated_reason"] == "max_episodes_reached"
    assert result.output["final_output"]["episodes_completed"] == 7


def test_default_trace_keeps_only_initial_and_final_policy_snapshots() -> None:
    class _CountingPolicy:
        def __init__(self) -> None:
            self.updates = 0
            self.get_params_calls = 0

        def select_action(self, state: RLState) -> str:
            return "good"

        def update(self, trajectory: Trajectory) -> dict[str, object]:
            self.updates += 1
            return {"updates": self.updates}

        def get_params(self) -> dict[str, object]:
            self.get_params_calls += 1
            return {"updates": self.updates}

    reset, step = _bandit_env()
    policy = _CountingPolicy()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        policy=policy,
        max_episodes=3,
        max_steps_per_episode=1,
    )
    result = pattern.run("learn", request_id="rl-test-snapshots")
    details = result.output["details"]

    assert policy.get_params_calls == 2
    assert details["initial_policy_params"] == {"updates": 0}
    assert "policy_params_history" not in details
    assert all("policy_params" not in trace for trace in details["episode_traces"])
    assert result.output["final_output"]["final_policy_params"] == {"updates": 3}
    assert result.output["workflow"]["step_results"]["rl_loop"]["output"]["iteration_results"] == []


def test_full_trace_keeps_one_policy_snapshot_per_episode() -> None:
    class _CountingPolicy:
        def __init__(self) -> None:
            self.updates = 0
            self.get_params_calls = 0

        def select_action(self, state: RLState) -> str:
            return "good"

        def update(self, trajectory: Trajectory) -> dict[str, object]:
            self.updates += 1
            return {"updates": self.updates}

        def get_params(self) -> dict[str, object]:
            self.get_params_calls += 1
            return {"updates": self.updates}

    reset, step = _bandit_env()
    policy = _CountingPolicy()
    result = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        policy=policy,
        max_episodes=3,
        max_steps_per_episode=1,
        trace_detail="full",
    ).run("learn", request_id="rl-test-full-snapshots")

    assert policy.get_params_calls == 4
    assert [trace["policy_params"] for trace in result.output["details"]["episode_traces"]] == [
        {"updates": 1},
        {"updates": 2},
        {"updates": 3},
    ]


def test_private_reward_stability_criterion_resets_after_reward_change() -> None:
    criterion = _RewardStabilityCriterion(threshold=0.1, stable_episodes=2)

    count, stopped = criterion.observe([1.0], 0)
    assert (count, stopped) == (0, False)
    count, stopped = criterion.observe([1.0, 1.05], count)
    assert (count, stopped) == (1, False)
    count, stopped = criterion.observe([1.0, 1.05, 1.08], count)
    assert (count, stopped) == (2, True)
    assert criterion.observe([1.0, 1.05, 1.08, 2.0], count) == (0, False)


def test_reaches_max_episodes_without_reward_stability_criterion() -> None:
    reset, step = _grid_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["left", "right", "stay"],
        max_episodes=3,
        max_steps_per_episode=10,
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
        random_seed=0,
    )

    result = pattern.run("cap", request_id="rl-test-cap")
    trace = result.output["details"]["episode_traces"][0]

    assert trace["steps"] == 4
    assert len(trace["step_traces"]) == 4
    assert trace["step_traces"][-1]["terminated"] is False
    assert trace["step_traces"][-1]["truncated"] is True


# ---------------------- Trace capture ----------------------


def test_traces_capture_states_actions_rewards_and_updates() -> None:
    reset, step = _grid_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["left", "right", "stay"],
        max_episodes=4,
        max_steps_per_episode=10,
        random_seed=7,
    )
    result = pattern.run("navigate", request_id="rl-test-trace")
    traces = result.output["details"]["episode_traces"]

    assert len(traces) == result.output["final_output"]["episodes_completed"]
    first = traces[0]
    assert set(first) >= {"episode", "episode_reward", "steps", "step_traces", "update_stats"}
    assert "policy_params" not in first
    assert first["steps"] >= 1

    first_step = first["step_traces"][0]
    assert set(first_step) >= {
        "step_num",
        "state",
        "action",
        "reward",
        "next_state",
        "terminated",
        "truncated",
        "done",
        "info",
    }
    assert "position" in first_step["state"]
    assert first_step["action"] in {"left", "right", "stay"}
    assert isinstance(first_step["reward"], float)
    assert "mean_return" in first["update_stats"]


def test_traces_are_isolated_from_in_place_mutating_environment() -> None:
    def reset() -> dict[str, object]:
        return {"position": {"value": 0}}

    def step(state: dict[str, object], action: str) -> tuple[dict[str, object], float, bool]:
        # Naive env: mutates a nested value in place and returns the same object.
        position = state["position"]
        assert isinstance(position, dict)
        position["value"] = int(position["value"]) + 1
        pos = int(position["value"])
        return state, (10.0 if pos == 3 else -1.0), pos == 3

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["step"],
        max_episodes=1,
        max_steps_per_episode=5,
        random_seed=0,
    )

    result = pattern.run("walk", request_id="rl-test-alias")
    steps_trace = result.output["details"]["episode_traces"][0]["step_traces"]

    # Each recorded pre-transition state must reflect the position at that step,
    # not the mutated end-of-episode value that in-place mutation would leak.
    positions = [int(trace["state"]["position"]["value"]) for trace in steps_trace]
    assert positions == [0, 1, 2]


def test_trace_snapshot_is_bounded_json_safe_and_best_effort() -> None:
    class _ArrayLike:
        def tolist(self) -> list[int]:
            return [1, 2]

    nested: object = "leaf"
    for _ in range(9):
        nested = [nested]

    snapshot = _trace_snapshot(
        {
            "long_text": "x" * 2_001,
            "large_mapping": {str(index): index for index in range(101)},
            "large_sequence": list(range(101)),
            "array_like": _ArrayLike(),
            "opaque": object(),
            "non_finite": float("nan"),
            "nested": nested,
        }
    )

    assert isinstance(snapshot, dict)
    assert snapshot["long_text"].endswith("<truncated:1>")
    assert snapshot["large_mapping"]["__truncated_items__"] == 1
    assert snapshot["large_sequence"][-1] == "<truncated-items:1>"
    assert snapshot["array_like"] == [1, 2]
    assert isinstance(snapshot["opaque"], str)
    assert snapshot["non_finite"] == "nan"
    assert "max-depth" in str(snapshot["nested"])


def test_state_normalization_rejects_non_mapping_and_non_string_keys() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        _normalize_state([("state", 1)], name="state")
    with pytest.raises(ValueError, match="keys must be strings"):
        _normalize_state({1: "state"}, name="state")


def test_trajectory_copy_boundaries_reject_invalid_or_uncopyable_values() -> None:
    class _Uncopyable:
        def __deepcopy__(self, memo: object) -> object:
            raise RuntimeError("cannot copy")

    class _CopyChangesShape(dict[str, object]):
        def __deepcopy__(self, memo: object) -> list[object]:
            return []

    with pytest.raises(TypeError, match="defensive copying"):
        ReinforcementLearningPattern._copy_state_for_trajectory({"value": _Uncopyable()})
    with pytest.raises(TypeError, match="copied state"):
        ReinforcementLearningPattern._copy_state_for_trajectory(_CopyChangesShape())
    with pytest.raises(TypeError, match="string or mapping"):
        ReinforcementLearningPattern._copy_action_for_trajectory(7)
    with pytest.raises(TypeError, match="copied action"):
        ReinforcementLearningPattern._copy_action_for_trajectory(_CopyChangesShape())


@pytest.mark.parametrize(
    ("reward", "done", "error"),
    [
        ("not-numeric", True, "reward must be numeric"),
        (float("inf"), True, "reward must be finite"),
        (1.0, 1, "done must be a bool"),
    ],
)
def test_environment_transition_validates_reward_and_done(
    reward: object,
    done: object,
    error: str,
) -> None:
    def reset() -> dict[str, object]:
        return {"state": 0}

    def step(state: RLState, action: RLAction) -> tuple[dict[str, object], float, bool]:
        return dict(state), reward, done  # type: ignore[return-value]

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["act"],
        max_episodes=1,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        pattern._step_environment({"state": 0}, "act")


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
        def select_action(self, state: RLState) -> RLAction:
            return {"force": 1.0}

        def update(self, trajectory: Trajectory) -> dict[str, object]:
            return {"mean_return": 0.0}

        def get_params(self) -> dict[str, object]:
            return {}

    def reset() -> dict[str, object]:
        return {"t": 0}

    def step(state: RLState, action: RLAction) -> tuple[dict[str, object], float, bool]:
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
        raise RuntimeError("environment boom")

    def step(state: RLState, action: RLAction) -> tuple[dict[str, object], float, bool]:
        return {"t": 1}, 0.0, True

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
