"""# Patterns / Reinforcement Learning Custom Policy.

## Introduction
Custom policies let the reinforcement learning pattern orchestrate state-dependent
or LLM-backed decisions without owning the learning algorithm. This example injects
an actor callable into a small feedback-guided policy. The deterministic actor keeps
the example reproducible; the same callable boundary can invoke an LLM that reads
the state and accumulated reward memory before selecting an action.


## Technical Implementation
1. Define a two-stage design environment where the best action changes by stage.
2. Implement the three-method structural policy contract: ``select_action``,
   ``update``, and ``get_params``.
3. Inject a deterministic actor that explores untried actions before exploiting the
   highest remembered reward; an LLM-backed actor can use the same inputs.
4. Run the public pattern and print the learned state-dependent strategy.

```mermaid
flowchart LR
    A["Design stage state"] --> B["Injected actor"]
    B --> C["Custom policy selects action"]
    C --> D["Environment returns reward"]
    D --> E["Policy updates reward memory"]
    E --> B
    E --> F["ExecutionResult with traces"]
```


## Expected Results

Output:

.. code-block:: text

   {
     "best_episode_reward": 5.0,
     "episodes_completed": 6,
     "learned_actions": {
       "concept": "explore",
       "detail": "refine"
     },
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "value_mode": "custom"
   }

## References
- `Reflexion: Language Agents with Verbal Reinforcement Learning <https://arxiv.org/abs/2303.11366>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Python typing protocols <https://typing.python.org/en/latest/spec/protocol.html>`_
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from design_research_agents import ReinforcementLearningPattern

State = Mapping[str, object]
Action = str | Mapping[str, object]
Trajectory = list[tuple[State, Action, float]]
Actor = Callable[[State, Mapping[str, Mapping[str, float]], Mapping[str, Mapping[str, int]]], str]
_ACTIONS = ("explore", "refine")


class FeedbackGuidedPolicy:
    """Small structural policy whose actor can be deterministic or LLM-backed."""

    def __init__(self, actor: Actor) -> None:
        """Store the injected actor and initialize empty reward memory."""
        self._actor = actor
        self._values: dict[str, dict[str, float]] = {}
        self._counts: dict[str, dict[str, int]] = {}

    def select_action(self, state: State) -> str:
        """Delegate action selection to the injected actor."""
        return self._actor(state, self._values, self._counts)

    def update(self, trajectory: Trajectory) -> dict[str, object]:
        """Update per-stage reward memory from one complete episode."""
        for state, action, reward in trajectory:
            if not isinstance(action, str):
                raise TypeError("FeedbackGuidedPolicy requires string actions.")
            stage = str(state["stage"])
            values, counts = self._tables(stage)
            counts[action] += 1
            values[action] += (reward - values[action]) / counts[action]
        return {"stages_updated": len({str(state["stage"]) for state, _, _ in trajectory})}

    def get_params(self) -> dict[str, object]:
        """Return fresh JSON-safe policy memory for traces and final output."""
        return {
            "values": {stage: dict(values) for stage, values in self._values.items()},
            "counts": {stage: dict(counts) for stage, counts in self._counts.items()},
            "learned_actions": {
                stage: max(values, key=values.get)
                for stage, values in self._values.items()
                if any(self._counts[stage].values())
            },
        }

    def _tables(self, stage: str) -> tuple[dict[str, float], dict[str, int]]:
        if stage not in self._values:
            self._values[stage] = {action: 0.0 for action in _ACTIONS}
            self._counts[stage] = {action: 0 for action in _ACTIONS}
        return self._values[stage], self._counts[stage]


def deterministic_actor(
    state: State,
    values_by_stage: Mapping[str, Mapping[str, float]],
    counts_by_stage: Mapping[str, Mapping[str, int]],
) -> str:
    """Explore each action once, then exploit remembered reward by design stage."""
    stage = str(state["stage"])
    values = values_by_stage.get(stage, {})
    counts = counts_by_stage.get(stage, {})
    for action in _ACTIONS:
        if counts.get(action, 0) == 0:
            return action
    return max(_ACTIONS, key=lambda action: values[action])


def main() -> None:
    """Learn a state-dependent design workflow through the custom-policy hook."""

    def environment_reset() -> dict[str, object]:
        return {"stage": "concept"}

    def environment_step(state: State, action: Action) -> tuple[dict[str, object], float, bool]:
        if not isinstance(action, str):
            raise TypeError("This environment requires a string action.")
        stage = str(state["stage"])
        if stage == "concept":
            return {"stage": "detail"}, (2.0 if action == "explore" else -1.0), False
        return {"stage": "complete"}, (3.0 if action == "refine" else -1.0), True

    pattern = ReinforcementLearningPattern(
        environment_reset=environment_reset,
        environment_step=environment_step,
        policy=FeedbackGuidedPolicy(deterministic_actor),
        max_episodes=6,
        max_steps_per_episode=2,
    )
    result = pattern.run(
        "Learn when to explore and when to refine a design.",
        request_id="example-pattern-reinforcement-learning-custom-policy-001",
    )

    final_output = result.output["final_output"]
    print(
        json.dumps(
            {
                "success": result.success,
                "best_episode_reward": final_output["best_episode_reward"],
                "episodes_completed": final_output["episodes_completed"],
                "terminated_reason": result.output["terminated_reason"],
                "value_mode": result.output["details"]["value_mode"],
                "learned_actions": final_output["final_policy_params"]["learned_actions"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
