"""# Patterns / Reinforcement Learning.

## Introduction
The built-in reinforcement learning policy is an honest global action-value
baseline. This example repeatedly selects one of three agent strategies for the
same benchmark family and learns which strategy receives the highest measured
reward. The seeded policy and deterministic benchmark make the result reproducible
without an LLM or machine-learning dependency.


## Technical Implementation
1. Define an environment reset delegate that identifies the benchmark family.
2. Define a one-step environment delegate that scores each selected agent strategy.
3. Execute ``ReinforcementLearningPattern.run(...)`` with discrete ``actions`` and
   no ``state_key``, selecting global-action value mode.
4. Print the learned action values and bounded training summary.

```mermaid
flowchart LR
    A["Benchmark task"] --> B["ReinforcementLearningPattern.run(...)"]
    B --> C["Epsilon-greedy strategy selection"]
    C --> D["Benchmark returns reward"]
    D --> E["Monte Carlo action-value update"]
    E --> C
    E --> F["ExecutionResult with traces"]
    F --> G["Printed JSON output"]
```


## Expected Results

Output:

.. code-block:: text

   {
     "action_values": {
       "critique_then_answer": 0.88,
       "direct_answer": 0.54,
       "tool_assisted": 0.73
     },
     "best_episode_reward": 0.88,
     "episodes_completed": 60,
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "value_mode": "global_action"
   }

## References
- `Sutton and Barto, Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
- `Multi-armed bandit algorithms and empirical evaluation <https://arxiv.org/abs/1003.0146>`_
- `AgentBench: Evaluating LLMs as Agents <https://arxiv.org/abs/2308.03688>`_
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents import ReinforcementLearningPattern

_STRATEGY_REWARDS = {
    "direct_answer": 0.54,
    "critique_then_answer": 0.88,
    "tool_assisted": 0.73,
}


def main() -> None:
    """Learn one global agent-strategy value table from deterministic rewards."""

    def environment_reset() -> dict[str, object]:
        return {"benchmark_family": "field_service_planning"}

    def environment_step(
        state: Mapping[str, object],
        action: str | Mapping[str, object],
    ) -> tuple[dict[str, object], float, bool]:
        if not isinstance(action, str):
            raise TypeError("This benchmark requires a discrete strategy name.")
        return dict(state), _STRATEGY_REWARDS[action], True

    pattern = ReinforcementLearningPattern(
        environment_reset=environment_reset,
        environment_step=environment_step,
        actions=list(_STRATEGY_REWARDS),
        max_episodes=60,
        max_steps_per_episode=1,
        gamma=1.0,
        epsilon=1.0,
        epsilon_decay=0.92,
        epsilon_min=0.05,
        random_seed=42,
    )
    result = pattern.run(
        "Learn which agent strategy performs best on the benchmark family.",
        request_id="example-pattern-reinforcement-learning-001",
    )

    final_output = result.output["final_output"]
    final_policy_params = final_output["final_policy_params"]
    print(
        json.dumps(
            {
                "success": result.success,
                "best_episode_reward": final_output["best_episode_reward"],
                "episodes_completed": final_output["episodes_completed"],
                "terminated_reason": result.output["terminated_reason"],
                "value_mode": final_policy_params["value_mode"],
                "action_values": final_policy_params["action_values"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
