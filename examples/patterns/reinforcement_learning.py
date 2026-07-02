"""# Patterns / Reinforcement Learning.

## Introduction
The reinforcement learning pattern runs an episodic agent-environment loop where
a policy learns from cumulative rewards. This example uses a simple grid navigation
environment with discrete actions. The agent starts at position 0 and learns to
move right to reach the goal at position 3. All delegates are deterministic so the
runtime contract is easy to inspect without an LLM dependency.


## Technical Implementation
1. Define an environment reset delegate that returns the initial state.
2. Define an environment step delegate that applies an action and returns the next
   state, reward, and done flag.
3. Execute ``ReinforcementLearningPattern.run(...)`` through the public patterns API.
4. Print a compact JSON payload showing the learned policy and reward history.

```mermaid
flowchart LR
    A["Initial state"] --> B["ReinforcementLearningPattern.run(...)"]
    B --> C["environment_reset starts each episode"]
    C --> D["EpsilonGreedyPolicy selects action"]
    D --> E["environment_step returns next_state, reward, done"]
    E --> F["Trajectory collected until done"]
    F --> G["Monte Carlo policy update"]
    G --> H["ExecutionResult/payload"]
    H --> I["Printed JSON output"]
```


## Expected Results

Example output shape:

.. code-block:: text

{
    "best_episode_reward": 7.0,
    "best_episode_index": ...,
    "episodes_completed": ...,
    "success": true,
    "terminated_reason": "converged"
}

## References
- `Reinforcement Learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_
- `Q-Learning <https://en.wikipedia.org/wiki/Q-learning>`_
- `Epsilon-greedy Algorithm <https://en.wikipedia.org/wiki/Epsilon-greedy_algorithm>`_
"""

from __future__ import annotations

import json

from design_research_agents import ReinforcementLearningPattern


def main() -> None:
    """Run one deterministic grid navigation RL workflow."""

    def environment_reset() -> dict[str, object]:
        return {"position": 0}

    def environment_step(
        state: dict[str, object],
        action: str,
    ) -> tuple[dict[str, object], float, bool]:
        pos = int(state["position"])
        if action == "right":
            pos += 1
        elif action == "left":
            pos -= 1
        reward = 10.0 if pos == 3 else -1.0
        done = pos == 3
        return {"position": pos}, reward, done

    pattern = ReinforcementLearningPattern(
        environment_reset=environment_reset,
        environment_step=environment_step,
        actions=["left", "right", "stay"],
        max_episodes=200,
        max_steps_per_episode=10,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.95,
        convergence_threshold=0.5,
        convergence_episodes=10,
        random_seed=42,
    )
    result = pattern.run(
        "Learn to navigate to position 3.",
        request_id="example-pattern-reinforcement-learning-001",
    )

    final_output = result.output["final_output"]
    print(
        json.dumps(
            {
                "success": result.success,
                "best_episode_reward": final_output["best_episode_reward"],
                "best_episode_index": final_output["best_episode_index"],
                "episodes_completed": final_output["episodes_completed"],
                "terminated_reason": result.output["terminated_reason"],
                "final_q_values": final_output.get("final_policy_params", {}).get("q_values"),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
