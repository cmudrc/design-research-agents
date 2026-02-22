"""Run traced ``NetworkedPattern`` and ``BlackboardPattern`` design coordination.

Expected observations:
- both peer-only patterns execute and report termination reasons.
- blackboard run reports message count and convergence status.
- ``trace`` metadata is present for both runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from _support_deterministic import FixedDesignPeerAgent

from design_research_agents import BlackboardPattern, NetworkedPattern, Tracer


def _summarize(result: object, request_id: str, tracer: Tracer) -> dict[str, object]:
    if not hasattr(result, "success") or not hasattr(result, "output"):
        return {"success": False, "error": "unexpected result", "trace": tracer.trace_info(request_id)}
    output = result.output if isinstance(result.output, dict) else {}
    blackboard = output.get("blackboard")
    messages = blackboard.get("messages") if isinstance(blackboard, dict) else []
    return {
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "rounds_executed": output.get("rounds_executed"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }


def main() -> None:
    """Run one networked and one blackboard coordination pass."""
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )

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

    print(
        json.dumps(
            {
                "networked_pattern": _summarize(network_result, network_request_id, tracer),
                "blackboard_pattern": _summarize(blackboard_result, blackboard_request_id, tracer),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
