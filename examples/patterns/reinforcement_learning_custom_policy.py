"""# Patterns / Reinforcement Learning With An Agent Policy.

## Introduction
An existing agent or workflow can act as a reinforcement-learning policy without
creating another orchestration pattern. This example places ``DirectLLMCall``
behind a small structural policy: the agent sees the current design stage,
available actions, and accumulated reward feedback, then selects one structured
action. A deterministic local client keeps the example offline and reproducible;
the same ``DirectLLMCall`` boundary can use any configured LLM client.


## Technical Implementation
1. Define a two-stage design environment where the best action changes by stage.
2. Wrap ``DirectLLMCall`` in a policy that prompts for one JSON action.
3. Learn from rich ``RLTransition`` objects through ``observe_transition`` and
   close each episode through ``end_episode``.
4. Train for six episodes, then evaluate the learned policy without updating it.

```mermaid
flowchart LR
    A["Design stage + reward memory"] --> B["DirectLLMCall policy actor"]
    B --> C["Structured explore or refine action"]
    C --> D["Environment transition"]
    D --> E["Policy observes RLTransition"]
    E --> A
    E --> F["Frozen evaluation"]
```


## Expected Results

Output:

.. code-block:: text

   {
     "agent_calls": 18,
     "best_episode_reward": 5.0,
     "episodes_completed": 6,
     "evaluation_mean_reward": 5.0,
     "learned_actions": {
       "concept": "explore",
       "detail": "refine"
     },
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "value_mode": "custom"
   }

## References
- `Agent Lightning: Train ANY AI Agents with Reinforcement Learning <https://arxiv.org/abs/2508.03680>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Python typing protocols <https://typing.python.org/en/latest/spec/protocol.html>`_
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import design_research_agents as drag

State = Mapping[str, object]
Action = str | Mapping[str, object]
_ACTIONS = ("explore", "refine")


class _DeterministicActionClient:
    """Offline LLM-client stand-in that follows the policy prompt exactly."""

    def generate(self, request: drag.LLMRequest) -> drag.LLMResponse:
        """Select an untried action during training, then exploit reward memory."""
        payload = json.loads(request.messages[-1].content)
        actions = tuple(str(action) for action in payload["actions"])
        values = {str(action): float(value) for action, value in payload["values"].items()}
        counts = {str(action): int(count) for action, count in payload["counts"].items()}

        action = next(
            (candidate for candidate in actions if payload["explore"] and counts.get(candidate, 0) == 0),
            max(actions, key=lambda candidate: (values.get(candidate, 0.0), -actions.index(candidate))),
        )
        return drag.LLMResponse(
            text=json.dumps({"action": action}, ensure_ascii=True, sort_keys=True),
            model=self.default_model(),
            provider="deterministic-policy-example",
        )

    def default_model(self) -> str:
        """Return the stable example model identifier."""
        return "deterministic-action-model"


class AgentFeedbackPolicy:
    """Transition-aware policy that delegates action selection to an agent."""

    def __init__(self, agent: drag.DirectLLMCall) -> None:
        """Store the actor and initialize per-stage reward memory."""
        self._agent = agent
        self._values: dict[str, dict[str, float]] = {}
        self._counts: dict[str, dict[str, int]] = {}
        self._agent_calls = 0

    def select_action(self, state: State) -> str:
        """Ask the agent to explore or exploit during training."""
        return self._ask_agent(state, explore=True)

    def select_evaluation_action(self, state: State) -> str:
        """Ask the agent for a greedy action during frozen evaluation."""
        return self._ask_agent(state, explore=False)

    def observe_transition(self, transition: drag.RLTransition) -> None:
        """Update reward memory immediately after one environment transition."""
        if not isinstance(transition.action, str):
            raise TypeError("AgentFeedbackPolicy requires string actions.")
        stage = str(transition.state["stage"])
        values, counts = self._tables(stage)
        action = transition.action
        counts[action] += 1
        values[action] += (transition.reward - values[action]) / counts[action]

    def end_episode(self, transitions: Sequence[drag.RLTransition]) -> dict[str, object]:
        """Return compact statistics after transition-level updates."""
        return {"stages_updated": len({str(transition.state["stage"]) for transition in transitions})}

    def get_params(self) -> dict[str, object]:
        """Return fresh JSON-safe policy memory for the final result."""
        return {
            "values": {stage: dict(values) for stage, values in self._values.items()},
            "counts": {stage: dict(counts) for stage, counts in self._counts.items()},
            "agent_calls": self._agent_calls,
            "learned_actions": {
                stage: max(values, key=values.get)
                for stage, values in self._values.items()
                if any(self._counts[stage].values())
            },
        }

    def _ask_agent(self, state: State, *, explore: bool) -> str:
        """Render policy context, execute the agent, and validate its action."""
        stage = str(state["stage"])
        values, counts = self._tables(stage)
        prompt = json.dumps(
            {
                "task": "Select exactly one action for the current design stage.",
                "stage": stage,
                "actions": _ACTIONS,
                "values": values,
                "counts": counts,
                "explore": explore,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        self._agent_calls += 1
        result = self._agent.run(
            prompt,
            request_id=f"example-rl-agent-policy-{self._agent_calls:03d}",
        )
        payload = json.loads(str(result.final_output))
        action = str(payload["action"])
        if action not in _ACTIONS:
            raise ValueError(f"Agent returned unsupported action {action!r}.")
        return action

    def _tables(self, stage: str) -> tuple[dict[str, float], dict[str, int]]:
        """Return initialized reward and visit tables for one design stage."""
        if stage not in self._values:
            self._values[stage] = {action: 0.0 for action in _ACTIONS}
            self._counts[stage] = {action: 0 for action in _ACTIONS}
        return self._values[stage], self._counts[stage]


def main() -> None:
    """Train and evaluate an agent-backed state-dependent policy."""

    def environment_reset() -> dict[str, object]:
        return {"stage": "concept"}

    def environment_step(state: State, action: Action) -> tuple[dict[str, object], float, bool]:
        if not isinstance(action, str):
            raise TypeError("This environment requires a string action.")
        stage = str(state["stage"])
        if stage == "concept":
            return {"stage": "detail"}, (2.0 if action == "explore" else -1.0), False
        return {"stage": "complete"}, (3.0 if action == "refine" else -1.0), True

    agent = drag.DirectLLMCall(
        llm_client=_DeterministicActionClient(),
        system_prompt="Return JSON containing one allowed action and no additional fields.",
        temperature=0.0,
    )
    policy = AgentFeedbackPolicy(agent)
    pattern = drag.ReinforcementLearningPattern(
        environment_reset=environment_reset,
        environment_step=environment_step,
        policy=policy,
        max_episodes=6,
        max_steps_per_episode=2,
        trace_detail="summary",
    )
    result = pattern.run(
        "Learn when to explore and when to refine a design.",
        request_id="example-pattern-reinforcement-learning-agent-policy-001",
    )
    evaluation = pattern.evaluate(episodes=3)

    final_output = result.output["final_output"]
    evaluated_policy_params = policy.get_params()
    print(
        json.dumps(
            {
                "success": result.success,
                "best_episode_reward": final_output["best_episode_reward"],
                "episodes_completed": final_output["episodes_completed"],
                "terminated_reason": result.output["terminated_reason"],
                "value_mode": result.output["details"]["value_mode"],
                "learned_actions": evaluated_policy_params["learned_actions"],
                "agent_calls": evaluated_policy_params["agent_calls"],
                "evaluation_mean_reward": evaluation.output["final_output"]["mean_reward"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
