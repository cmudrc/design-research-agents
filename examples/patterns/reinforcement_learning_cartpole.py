"""# Patterns / Reinforcement Learning CartPole.

## Introduction
This example learns when to push left or right in Gymnasium's ``CartPole-v1``
inverted-pendulum environment. Unlike controller selection, every
reinforcement-learning action applies one force for one simulation timestep.
The built-in tabular Monte Carlo policy learns state-action values from complete
episodes without a preprogrammed balancing controller.


## Technical Implementation
From a source checkout, install the optional environment dependency with
``python -m pip install -e ".[rl]"``. The published-package equivalent is
``python -m pip install "design-research-agents[rl]"``. Then:

1. Reset CartPole with a different deterministic seed for each training episode.
2. Discretize pole angle and pole angular velocity into a compact state key.
3. Let ``ReinforcementLearningPattern`` choose ``push_left`` or ``push_right``
   at every simulation timestep.
4. Update tabular :math:`Q(s, a)` estimates from each complete episode return.
5. Evaluate the greedy learned policy on 50 held-out seeded initial states.

```mermaid
flowchart LR
    A["Observe CartPole state"] --> B["Discretize angle and angular velocity"]
    B --> C["Select left or right force"]
    C --> D["Advance Gymnasium one timestep"]
    D --> E{"Failure or 200 steps?"}
    E -->|Continue| A
    E -->|Episode done| F["Update tabular state-action values"]
    F --> A
```


## Expected Results

Output:

.. code-block:: text

   {
     "episodes_completed": 100,
     "evaluation_episodes": 50,
     "evaluation_mean_steps": 200.0,
     "evaluation_min_steps": 200,
     "first_20_mean_steps": 46.9,
     "last_20_mean_steps": 200.0,
     "learned_state_bins": 91,
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "unseen_evaluation_states": 0,
     "value_mode": "state_action"
   }

## References
- `Gymnasium CartPole environment <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_
- `Gymnasium basic usage <https://gymnasium.farama.org/introduction/basic_usage/>`_
- `Sutton and Barto, Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Mapping, Sequence

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - exercised by base-install users
    raise SystemExit(
        'Install with python -m pip install "design-research-agents[rl]", '
        'or use python -m pip install -e ".[rl]" from a source checkout.'
    ) from exc

from design_research_agents import ReinforcementLearningPattern

_MAX_BALANCE_STEPS = 200
_TRAINING_EPISODES = 100
_EVALUATION_EPISODES = 50
_ACTIONS = ("push_left", "push_right")
_ACTION_INDEX = {"push_left": 0, "push_right": 1}
_OBSERVATION_FIELDS = (
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
)
_ANGLE_THRESHOLDS = (-0.16, -0.10, -0.06, -0.025, 0.0, 0.025, 0.06, 0.10, 0.16)
_ANGULAR_VELOCITY_THRESHOLDS = (-3.0, -1.5, -0.75, -0.35, -0.12, 0.0, 0.12, 0.35, 0.75, 1.5, 3.0)


def _state_from_observation(observation: Sequence[float]) -> dict[str, object]:
    """Convert one CartPole observation into the pattern's mapping state."""
    return {name: float(observation[index]) for index, name in enumerate(_OBSERVATION_FIELDS)}


def _state_key(state: Mapping[str, object]) -> str:
    """Discretize the two pole variables that dominate balancing decisions."""
    angle_bin = bisect_left(_ANGLE_THRESHOLDS, float(state["pole_angle"]))
    angular_velocity_bin = bisect_left(
        _ANGULAR_VELOCITY_THRESHOLDS,
        float(state["pole_angular_velocity"]),
    )
    return f"angle={angle_bin}:angular_velocity={angular_velocity_bin}"


def _greedy_action(state: Mapping[str, object], q_values: Mapping[str, object]) -> tuple[str, bool]:
    """Select the best learned force and report whether its state was unseen."""
    raw_values = q_values.get(_state_key(state))
    values = raw_values if isinstance(raw_values, Mapping) else {}
    action = max(
        _ACTIONS,
        key=lambda candidate: (float(values.get(candidate, 0.0)), -_ACTIONS.index(candidate)),
    )
    return action, not values


def main() -> None:
    """Learn and evaluate left/right forces against CartPole dynamics."""
    environment = gym.make("CartPole-v1", max_episode_steps=_MAX_BALANCE_STEPS)
    episode_index = 0

    def environment_reset() -> dict[str, object]:
        nonlocal episode_index
        observation, _ = environment.reset(seed=10_000 + episode_index)
        episode_index += 1
        return _state_from_observation(observation)

    def environment_step(
        state: Mapping[str, object],
        action: str | Mapping[str, object],
    ) -> tuple[dict[str, object], float, bool]:
        del state
        if not isinstance(action, str):
            raise TypeError("CartPole requires a discrete left or right force.")
        observation, reward, terminated, truncated, _ = environment.step(_ACTION_INDEX[action])
        return _state_from_observation(observation), float(reward), bool(terminated or truncated)

    try:
        pattern = ReinforcementLearningPattern(
            environment_reset=environment_reset,
            environment_step=environment_step,
            actions=_ACTIONS,
            state_key=_state_key,
            max_episodes=_TRAINING_EPISODES,
            max_steps_per_episode=_MAX_BALANCE_STEPS,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.94,
            epsilon_min=0.03,
            random_seed=42,
        )
        result = pattern.run(
            "Learn left and right forces that keep the inverted pendulum upright.",
            request_id="example-pattern-reinforcement-learning-cartpole-001",
        )

        final_output = result.output["final_output"]
        final_policy_params = final_output["final_policy_params"]
        q_values = final_policy_params["q_values"]
        episode_rewards = final_output["episode_rewards"]

        evaluation_steps: list[int] = []
        unseen_evaluation_states = 0
        for seed in range(20_000, 20_000 + _EVALUATION_EPISODES):
            observation, _ = environment.reset(seed=seed)
            state = _state_from_observation(observation)
            balanced_steps = 0
            for _ in range(_MAX_BALANCE_STEPS):
                action, unseen = _greedy_action(state, q_values)
                unseen_evaluation_states += int(unseen)
                observation, _, terminated, truncated, _ = environment.step(_ACTION_INDEX[action])
                state = _state_from_observation(observation)
                balanced_steps += 1
                if terminated or truncated:
                    break
            evaluation_steps.append(balanced_steps)
    finally:
        environment.close()

    print(
        json.dumps(
            {
                "success": result.success,
                "episodes_completed": final_output["episodes_completed"],
                "terminated_reason": result.output["terminated_reason"],
                "value_mode": final_policy_params["value_mode"],
                "first_20_mean_steps": round(sum(episode_rewards[:20]) / 20, 1),
                "last_20_mean_steps": round(sum(episode_rewards[-20:]) / 20, 1),
                "learned_state_bins": len(q_values),
                "evaluation_episodes": len(evaluation_steps),
                "evaluation_mean_steps": round(sum(evaluation_steps) / len(evaluation_steps), 1),
                "evaluation_min_steps": min(evaluation_steps),
                "unseen_evaluation_states": unseen_evaluation_states,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
