"""Run traced ``NetworkedPattern`` and ``BlackboardPattern`` design coordination.

Expected observations:
- both peer-only patterns execute and report termination reasons.
- blackboard run reports message count and convergence status.
- ``trace`` metadata is present for both runs.
"""

from __future__ import annotations

from design_research_agents import BlackboardPattern, NetworkedPattern
from design_research_agents._shared._deterministic_design_helpers import FixedDesignPeerAgent
from design_research_agents._shared._example_support import make_tracer, print_json, trace_info


def _summarize(result: object, request_id: str) -> dict[str, object]:
    if not hasattr(result, "success") or not hasattr(result, "output"):
        return {"success": False, "error": "unexpected result", "trace": trace_info(request_id)}
    output = result.output if isinstance(result.output, dict) else {}
    blackboard = output.get("blackboard")
    messages = blackboard.get("messages") if isinstance(blackboard, dict) else []
    return {
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "rounds_executed": output.get("rounds_executed"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "final_output": output.get("final_output"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }


def main() -> None:
    """Run one networked and one blackboard coordination pass."""
    tracer = make_tracer()

    network_request_id = "example-workflow-networked-pattern-design-001"
    networked = NetworkedPattern(
        peers={
            "peer_b": FixedDesignPeerAgent(messages=["peer_b proposes modular latch"]),
            "peer_a": FixedDesignPeerAgent(messages=["peer_a proposes captive screw rail"]),
        },
        max_rounds=2,
        tracer=tracer,
    )
    network_result = networked.run(
        "Coordinate candidate mechanisms for a field-serviceable sensor enclosure.",
        request_id=network_request_id,
    )

    blackboard_request_id = "example-workflow-blackboard-pattern-design-001"
    blackboard = BlackboardPattern(
        peers={
            "peer_b": FixedDesignPeerAgent(messages=["peer_b proposes option B"]),
            "peer_a": FixedDesignPeerAgent(messages=["peer_a proposes option A"]),
        },
        max_rounds=3,
        stability_rounds=2,
        tracer=tracer,
    )
    blackboard_result = blackboard.run(
        "Compare two concept options and converge on a serviceable design direction.",
        request_id=blackboard_request_id,
    )

    print_json(
        {
            "networked_pattern": _summarize(network_result, network_request_id),
            "blackboard_pattern": _summarize(blackboard_result, blackboard_request_id),
        }
    )


if __name__ == "__main__":
    main()
