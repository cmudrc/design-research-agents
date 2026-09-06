"""# Agents / Paper Contributions.

## Introduction
Agent studies need a compact writing-support boundary that records configuration without
mistaking availability for execution. This offline example describes a tracer before any
run evidence exists, so the resulting packet keeps the missing trace visible as a gap.


## Technical Implementation
1. Configure a disabled ``Tracer`` without starting an agent run.
2. Call ``collect_agent_paper_contributions(...)`` with a stable component identifier.
3. Verify the public paper-contribution contract version and print the JSON-compatible packet.

```mermaid
flowchart LR
    A["Configured tracer"] --> B["collect_agent_paper_contributions(...)"]
    B --> C["Configured Methods contribution"]
    B --> D["Execution evidence gap"]
    C --> E["Versioned component packet"]
    D --> E
```


## Expected Results

The printed packet contains a configured Methods contribution, package provenance, and an
``execution-not-provided`` gap. It contains no observed execution claim.

## References
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `The ACM Artifact Review and Badging policy <https://www.acm.org/publications/policies/artifact-review-and-badging-current>`_
- `FAIR Guiding Principles for scientific data management <https://doi.org/10.1038/sdata.2016.18>`_
"""

from __future__ import annotations

import json

import design_research_agents as drag


def main() -> None:
    """Print deterministic writing support for one configured component."""
    tracer = drag.Tracer(enabled=False, enable_jsonl=False, enable_console=False)
    packet = drag.collect_agent_paper_contributions(
        tracer,
        component_id="example.tracer",
    )
    assert packet["schema_version"] == drag.PAPER_CONTRIBUTION_VERSION
    print(json.dumps(packet, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
