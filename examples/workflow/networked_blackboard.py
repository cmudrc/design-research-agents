"""Runnable example for ``BlackboardPattern`` peer coordination."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.workflow import BlackboardPattern


class FixedPeerAgent(Agent):
    """Deterministic peer used for blackboard demonstration."""

    def __init__(self, *, message: str, stop: bool = False) -> None:
        """Store fixed contribution payload for this peer.

        Args:
            message: Message this peer contributes each round.
            stop: Whether this peer signals stop.
        """
        self._message = message
        self._stop = stop

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Return one fixed peer contribution.

        Args:
            prompt: Peer prompt payload.
            request_id: Optional request identifier.
            dependencies: Optional dependency mapping.

        Returns:
            Deterministic peer contribution.
        """
        del prompt, request_id, dependencies
        return ExecutionResult(
            output={
                "messages": [self._message],
                "proposals": {},
                "decisions": {},
                "stop": self._stop,
            },
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"peer": self._message},
        )


def main() -> None:
    """Run one blackboard coordination workflow and print result."""
    pattern = BlackboardPattern(
        peers={
            "peer_b": FixedPeerAgent(message="peer_b proposes option B"),
            "peer_a": FixedPeerAgent(message="peer_a proposes option A"),
        },
        max_rounds=3,
        stability_rounds=2,
    )
    result = pattern.run("Compare two concept options and converge.")
    print(result.asdict())


if __name__ == "__main__":
    main()
