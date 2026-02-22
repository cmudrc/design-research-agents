r"""# Patterns / Networked Blackboard.

## Introduction
Blackboard-system architecture motivates shared-state collaboration among specialized problem solvers,
AutoGen informs practical multi-agent implementation choices, and Human-AI collaboration by design clarifies
governance value in shared workspace reasoning. This example builds a networked blackboard pattern with
explicit execution records.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``NetworkedPattern.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["NetworkedPattern.run(...)"]
    C --> D["blackboard workers contribute and aggregate shared state"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "blackboard_pattern": {
       "error": null,
       "example": "patterns/networked_blackboard.py",
       "final_output": {
         "blackboard": {
           "decisions": {
             "peer_a": {},
             "peer_b": {}
           },
           "history": [
             {
               "contributions": {
                 "peer_a": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: standardize fastener head geomet...
                   ],
                   "peer_id": "peer_a",
                   "proposals": {},
                   "round": 1,
                   "stop": false
                 },
                 "peer_b": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: preserve ingress protection whil...
                   ],
                   "peer_id": "peer_b",
                   "proposals": {},
                   "round": 1,
                   "stop": false
                 }
               },
               "round": 1
             },
             {
               "contributions": {
                 "peer_a": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: compare option A and option B ma...
                   ],
                   "peer_id": "peer_a",
                   "proposals": {},
                   "round": 2,
                   "stop": false
                 },
                 "peer_b": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: select the concept with fastest ...
                   ],
                   "peer_id": "peer_b",
                   "proposals": {},
                   "round": 2,
                   "stop": false
                 }
               },
               "round": 2
             },
             {
               "contributions": {
                 "peer_a": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: document final maintenance SOP c...
                   ],
                   "peer_id": "peer_a",
                   "proposals": {},
                   "round": 3,
                   "stop": false
                 },
                 "peer_b": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: finalize blackboard recommendati...
                   ],
                   "peer_id": "peer_b",
                   "proposals": {},
                   "round": 3,
                   "stop": false
                 }
               },
               "round": 3
             }
           ],
           "messages": [
             {
               "message": "{"model": "example-model", "model_text": "Peer contribution: standardize fastener head g...
               "peer_id": "peer_a",
               "round": 1
             },
             {
               "message": "{"model": "example-model", "model_text": "Peer contribution: preserve ingress protection...
               "peer_id": "peer_b",
               "round": 1
             },
             {
               "message": "{"model": "example-model", "model_text": "Peer contribution: compare option A and option...
               "peer_id": "peer_a",
               "round": 2
             },
             {
               "message": "{"model": "example-model", "model_text": "Peer contribution: select the concept with fas...
               "peer_id": "peer_b",
               "round": 2
             },
             {
               "message": "{"model": "example-model", "model_text": "Peer contribution: document final maintenance ...
               "peer_id": "peer_a",
               "round": 3
             },
             {
               "message": "{"model": "example-model", "model_text": "Peer contribution: finalize blackboard recomme...
               "peer_id": "peer_b",
               "round": 3
             }
           ],
           "proposals": {
             "peer_a": {},
             "peer_b": {}
           },
           "round": 3,
           "state_hash": "37d9c4f80222a85aca8f3e97a0eccc0a6df5f393ff5519d24bc64b15f1d35e17",
           "task": "Compare two concept options and converge on a serviceable design direction."
         },
         "rounds_executed": 3,
         "terminated_reason": "max_rounds_reached"
       },
       "message_count": 6,
       "rounds_executed": 3,
       "success": true,
       "terminated_reason": "max_rounds_reached",
       "trace": {
         "request_id": "example-workflow-blackboard-pattern-design-001",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     },
     "networked_pattern": {
       "error": null,
       "example": "patterns/networked_blackboard.py",
       "final_output": {
         "blackboard": {
           "decisions": {},
           "history": [
             {
               "contributions": {
                 "peer_a": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: prioritize captive screws for qu...
                   ],
                   "peer_id": "peer_a",
                   "proposals": {},
                   "round": 1,
                   "stop": false
                 },
                 "peer_b": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: keep gasket alignment features f...
                   ],
                   "peer_id": "peer_b",
                   "proposals": {},
                   "round": 1,
                   "stop": false
                 }
               },
               "round": 1
             },
             {
               "contributions": {
                 "peer_a": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: propose tool-less battery tray r...
                   ],
                   "peer_id": "peer_a",
                   "proposals": {},
                   "round": 2,
                   "stop": false
                 },
                 "peer_b": {
                   "decisions": {},
                   "messages": [
                     "{"model": "example-model", "model_text": "Peer contribution: add visual fastener indexing for...
                   ],
                   "peer_id": "peer_b",
                   "proposals": {},
                   "round": 2,
                   "stop": false
                 }
               },
               "round": 2
             }
           ],
           "messages": [],
           "proposals": {},
           "round": 0,
           "state_hash": "81061604424f0da3273856c2539bf4d58787542d5742d019e7dcadd3352b33f8",
           "task": "Coordinate candidate mechanisms for a field-serviceable sensor enclosure."
         },
         "rounds_executed": 2,
         "terminated_reason": "max_rounds_reached"
       },
       "message_count": 0,
       "rounds_executed": 2,
       "success": true,
       "terminated_reason": "max_rounds_reached",
       "trace": {
         "request_id": "example-workflow-networked-pattern-design-001",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     }
   }


## References
- `Blackboard System (Wikipedia) <https://en.wikipedia.org/wiki/Blackboard_system>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DirectLLMCall,
    ExecutionResult,
    LlamaCppServerLLMClient,
    Tracer,
)
from design_research_agents.patterns import BlackboardPattern, NetworkedPattern


def _summarize(result: ExecutionResult) -> dict[str, object]:
    blackboard = result.output_dict("blackboard")
    messages = blackboard.get("messages") if isinstance(blackboard, dict) else []
    return result.summary(
        details={
            "rounds_executed": result.output_value("rounds_executed"),
            "message_count": len(messages) if isinstance(messages, list) else 0,
        },
    )


def main() -> None:
    """Run one networked and one blackboard coordination pass."""
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )

    llm_client = LlamaCppServerLLMClient()
    try:
        peer_a = DirectLLMCall(llm_client=llm_client, tracer=tracer)
        peer_b = DirectLLMCall(llm_client=llm_client, tracer=tracer)

        network_request_id = "example-workflow-networked-pattern-design-001"
        networked = NetworkedPattern(
            peers={
                "peer_b": peer_b,
                "peer_a": peer_a,
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
                "peer_b": peer_b,
                "peer_a": peer_a,
            },
            max_rounds=3,
            stability_rounds=2,
            tracer=tracer,
        )
        blackboard_result = blackboard.run(
            "Compare two concept options and converge on a serviceable design direction.",
            request_id=blackboard_request_id,
        )
    finally:
        llm_client.close()

    print(
        json.dumps(
            {
                "networked_pattern": _summarize(network_result),
                "blackboard_pattern": _summarize(blackboard_result),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
