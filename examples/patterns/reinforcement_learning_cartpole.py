"""# Patterns / Reinforcement Learning CartPole.

## Introduction
This example learns which feedback-controller design balances Gymnasium's
``CartPole-v1`` inverted pendulum most reliably. Each reinforcement-learning
action selects one complete controller, and the environment returns the number
of balanced simulation steps as reward. The formulation is a transparent
controller-selection problem, not per-timestep Q-learning.


## Technical Implementation
From a source checkout, install the optional environment dependency with
``python -m pip install -e ".[rl]"``. The published-package equivalent is
``python -m pip install "design-research-agents[rl]"``. Then:

1. Reset CartPole with a different deterministic seed for each training episode.
2. Let ``ReinforcementLearningPattern`` select one candidate feedback controller.
3. Apply that controller's left/right force decisions for at most 200 simulation
   steps as one macro action.
4. Learn global action values from the number of balanced steps.
5. Evaluate the best learned controller on 25 held-out seeded initial states.

```mermaid
flowchart LR
    A["Seeded CartPole reset"] --> B["Select feedback controller"]
    B --> C["Apply left or right force"]
    C --> D{"Failure or 200 steps?"}
    D -->|Continue| C
    D -->|Done| E["Reward equals balanced steps"]
    E --> F["Update controller value"]
    F --> B
```


## Expected Results

Output:

.. code-block:: text

   {
     "action_values": {
       "angle_only": 48.6,
       "full_state": 149.8,
       "overcorrected": 189.29,
       "rate_assisted": 200.0
     },
     "episodes_completed": 40,
     "evaluation_mean_steps": 200.0,
     "evaluation_min_steps": 200,
     "learned_controller": "rate_assisted",
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "value_mode": "global_action"
   }

## References
- `Gymnasium CartPole environment <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_
- `Gymnasium basic usage <https://gymnasium.farama.org/introduction/basic_usage/>`_
- `Sutton and Barto, Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
"""

from __future__ import annotations

import json
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
_TRAINING_EPISODES = 40
_ACTION_TO_FORCE = {False: 0, True: 1}
_OBSERVATION_FIELDS = (
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
)
_CONTROLLER_GAINS = {
    "angle_only": (0.0, 0.0, 1.0, 0.0),
    "rate_assisted": (0.0, 0.0, 1.0, 0.15),
    "full_state": (-0.20, -0.35, 1.0, 0.18),
    "overcorrected": (-0.10, -0.20, 1.0, 0.30),
}


def _state_from_observation(
    observation: Sequence[float],
    *,
    seed: int,
    balanced_steps: int = 0,
) -> dict[str, object]:
    """Convert one CartPole observation into the pattern's mapping state."""
    return {
        **{name: float(observation[index]) for index, name in enumerate(_OBSERVATION_FIELDS)},
        "seed": seed,
        "balanced_steps": balanced_steps,
    }


def _force_for_controller(observation: Sequence[float], gains: Sequence[float]) -> int:
    """Return Gymnasium's left/right force index from linear state feedback."""
    push_right = sum(float(value) * gain for value, gain in zip(observation, gains, strict=True)) > 0
    return _ACTION_TO_FORCE[push_right]


def main() -> None:
    """Learn and evaluate a feedback controller against CartPole dynamics."""
    environment = gym.make("CartPole-v1", max_episode_steps=_MAX_BALANCE_STEPS)
    episode_index = 0

    def run_controller(
        observation: Sequence[float],
        gains: Sequence[float],
    ) -> tuple[Sequence[float], int]:
        balanced_steps = 0
        for _ in range(_MAX_BALANCE_STEPS):
            observation, _, terminated, truncated, _ = environment.step(_force_for_controller(observation, gains))
            balanced_steps += 1
            if terminated or truncated:
                break
        return observation, balanced_steps

    def environment_reset() -> dict[str, object]:
        nonlocal episode_index
        seed = 10_000 + episode_index
        episode_index += 1
        observation, _ = environment.reset(seed=seed)
        return _state_from_observation(observation, seed=seed)

    def environment_step(
        state: Mapping[str, object],
        action: str | Mapping[str, object],
    ) -> tuple[dict[str, object], float, bool]:
        if not isinstance(action, str):
            raise TypeError("CartPole controller selection requires a controller name.")

        gains = _CONTROLLER_GAINS[action]
        observation = tuple(float(state[name]) for name in _OBSERVATION_FIELDS)
        observation, balanced_steps = run_controller(observation, gains)

        next_state = _state_from_observation(
            observation,
            seed=int(state["seed"]),
            balanced_steps=balanced_steps,
        )
        return next_state, float(balanced_steps), True

    try:
        pattern = ReinforcementLearningPattern(
            environment_reset=environment_reset,
            environment_step=environment_step,
            actions=tuple(_CONTROLLER_GAINS),
            max_episodes=_TRAINING_EPISODES,
            max_steps_per_episode=1,
            gamma=1.0,
            epsilon=1.0,
            epsilon_decay=0.94,
            epsilon_min=0.1,
            random_seed=42,
        )
        result = pattern.run(
            "Learn which feedback controller keeps the inverted pendulum upright.",
            request_id="example-pattern-reinforcement-learning-cartpole-001",
        )

        final_output = result.output["final_output"]
        final_policy_params = final_output["final_policy_params"]
        action_values = final_policy_params["action_values"]
        learned_controller = max(_CONTROLLER_GAINS, key=action_values.__getitem__)

        evaluation_steps: list[int] = []
        for seed in range(20_000, 20_025):
            observation, _ = environment.reset(seed=seed)
            _, balanced_steps = run_controller(observation, _CONTROLLER_GAINS[learned_controller])
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
                "action_values": {name: round(action_values[name], 2) for name in _CONTROLLER_GAINS},
                "learned_controller": learned_controller,
                "evaluation_mean_steps": round(sum(evaluation_steps) / len(evaluation_steps), 1),
                "evaluation_min_steps": min(evaluation_steps),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
