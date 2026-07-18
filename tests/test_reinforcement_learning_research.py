"""Research-facing transition, lifecycle, and evaluation tests."""

from __future__ import annotations

import pytest

from design_research_agents._implementations import ReinforcementLearningPattern
from design_research_agents._implementations._patterns._reinforcement_learning_pattern import (
    EnvironmentResetDelegate,
    EnvironmentStepDelegate,
    RLAction,
    RLState,
    RLTransition,
    Trajectory,
)


def _bandit_env() -> tuple[EnvironmentResetDelegate, EnvironmentStepDelegate]:
    def reset() -> dict[str, object]:
        return {"state": 0}

    def step(state: RLState, action: RLAction) -> tuple[dict[str, object], float, bool]:
        return dict(state), (1.0 if action == "good" else -1.0), True

    return reset, step


def test_gymnasium_style_transition_preserves_termination_and_info() -> None:
    def reset() -> dict[str, object]:
        return {"state": 0}

    def step(
        state: RLState,
        action: RLAction,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        return {"state": 1}, 2.5, False, True, {"limit": "time"}

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["act"],
        max_episodes=1,
    )
    result = pattern.run("act", request_id="rl-test-rich-transition")
    transition_trace = result.output["details"]["episode_traces"][0]["step_traces"][0]

    assert transition_trace["terminated"] is False
    assert transition_trace["truncated"] is True
    assert transition_trace["done"] is True
    assert transition_trace["info"] == {"limit": "time"}


def test_environment_transition_validates_five_value_shape() -> None:
    def reset() -> dict[str, object]:
        return {"state": 0}

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=lambda state, action: (  # type: ignore[arg-type,return-value]
            {"state": 1},
            1.0,
            False,
            False,
            [],
        ),
        actions=["act"],
        max_episodes=1,
    )
    with pytest.raises(TypeError, match="info must be a mapping"):
        pattern._step_environment({"state": 0}, "act")


def test_transition_aware_policy_observes_each_step_and_ends_each_episode() -> None:
    class _OnlinePolicy:
        def __init__(self) -> None:
            self.transitions: list[RLTransition] = []
            self.episodes = 0

        def select_action(self, state: RLState) -> str:
            return "advance"

        def observe_transition(self, transition: RLTransition) -> None:
            self.transitions.append(transition)

        def end_episode(self, transitions: tuple[RLTransition, ...]) -> dict[str, object]:
            self.episodes += 1
            assert tuple(self.transitions[-len(transitions) :]) == transitions
            return {"observed_transitions": len(transitions)}

        def get_params(self) -> dict[str, object]:
            return {"episodes": self.episodes}

    def reset() -> dict[str, object]:
        return {"position": 0}

    def step(
        state: RLState,
        action: RLAction,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        position = int(state["position"]) + 1
        return {"position": position}, 1.0, position == 2, False, {"position": position}

    policy = _OnlinePolicy()
    result = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        policy=policy,
        max_episodes=2,
        max_steps_per_episode=3,
    ).run("advance", request_id="rl-test-transition-policy")

    assert policy.episodes == 2
    assert len(policy.transitions) == 4
    assert all(isinstance(transition, RLTransition) for transition in policy.transitions)
    assert policy.transitions[0].next_state == {"position": 1}
    assert policy.transitions[1].terminated is True
    assert result.output["details"]["episode_traces"][0]["update_stats"] == {"observed_transitions": 2}


def test_transition_aware_policy_requires_complete_hook_pair() -> None:
    class _IncompletePolicy:
        def select_action(self, state: RLState) -> str:
            return "act"

        def observe_transition(self, transition: RLTransition) -> None:
            return None

        def get_params(self) -> dict[str, object]:
            return {}

    reset, step = _bandit_env()
    with pytest.raises(ValueError, match="both observe_transition and end_episode"):
        ReinforcementLearningPattern(
            environment_reset=reset,
            environment_step=step,
            policy=_IncompletePolicy(),  # type: ignore[arg-type]
        )


def test_evaluate_uses_frozen_greedy_policy_and_separate_environment() -> None:
    training_reset, training_step = _bandit_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=training_reset,
        environment_step=training_step,
        actions=["good", "bad"],
        max_episodes=4,
        max_steps_per_episode=1,
        epsilon=0.0,
        random_seed=0,
        trace_detail="summary",
    )
    pattern.run("learn", request_id="rl-test-evaluate-train")
    params_before = pattern._policy.get_params()

    reset_calls = 0

    def evaluation_reset() -> dict[str, object]:
        nonlocal reset_calls
        reset_calls += 1
        return {"evaluation": reset_calls}

    def evaluation_step(
        state: RLState,
        action: RLAction,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        return dict(state), (1.0 if action == "good" else -1.0), True, False, {"split": "held-out"}

    result = pattern.evaluate(
        episodes=3,
        environment_reset=evaluation_reset,
        environment_step=evaluation_step,
        request_id="rl-test-evaluate",
    )

    assert result.success
    assert result.output["terminated_reason"] == "evaluation_completed"
    assert result.output["final_output"]["episode_rewards"] == [1.0, 1.0, 1.0]
    assert result.output["final_output"]["mean_steps"] == 1.0
    assert result.output["details"]["trace_detail"] == "summary"
    assert "step_traces" not in result.output["details"]["episode_traces"][0]
    assert pattern._policy.get_params() == params_before


def test_tabular_evaluation_does_not_add_unseen_states() -> None:
    def reset() -> dict[str, object]:
        return {"stage": "train"}

    def step(state: RLState, action: RLAction) -> tuple[dict[str, object], float, bool]:
        return dict(state), 1.0, True

    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        actions=["act"],
        state_key=lambda state: str(state["stage"]),
        max_episodes=1,
        epsilon=0.0,
    )
    pattern.run("learn")
    params_before = pattern._policy.get_params()

    pattern.evaluate(
        episodes=1,
        environment_reset=lambda: {"stage": "held-out"},
    )

    assert pattern._policy.get_params() == params_before


def test_custom_policy_evaluation_requires_explicit_selector() -> None:
    class _CustomPolicy:
        def select_action(self, state: RLState) -> str:
            return "good"

        def update(self, trajectory: Trajectory) -> dict[str, object]:
            return {}

        def get_params(self) -> dict[str, object]:
            return {}

    reset, step = _bandit_env()
    pattern = ReinforcementLearningPattern(
        environment_reset=reset,
        environment_step=step,
        policy=_CustomPolicy(),
        max_episodes=1,
    )

    with pytest.raises(ValueError, match="action_selector"):
        pattern.evaluate(episodes=1)
    result = pattern.evaluate(episodes=1, action_selector=lambda state: "good")
    assert result.output["final_output"]["mean_reward"] == 1.0
