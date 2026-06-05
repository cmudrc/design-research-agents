"""# Workflow / Demo JSON Tool Workflow.

## Introduction
This workshop-sized example uses ``DemoLLMClient`` with a JSON-mode
``MultiStepAgent`` so Qwen3-0.6B can select a real core tool and then finish
with a structured answer.

## Technical Implementation
1. Create a ``Toolbox`` with core tools and a ``DemoLLMClient`` using Qwen3-0.6B
   GGUF defaults.
2. Configure ``MultiStepAgent(mode="json")`` with ``text.word_count`` as the
   only allowed runtime tool.
3. Run a short prompt that forces one tool call and one final-answer step, then
   print the normalized execution summary.

## Expected Results
The example prints an ``ExecutionResult.summary()`` payload. Under deterministic
example tests, the model calls are monkeypatched to avoid starting llama.cpp.

## References
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `Toolformer <https://arxiv.org/abs/2302.04761>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag


def main() -> None:
    """Run a JSON tool workflow through the Qwen3 demo client."""
    request_id = "example-workflow-demo-json-tool-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    with drag.Toolbox() as tool_runtime, drag.DemoLLMClient(name="demo-qwen3-workflow") as llm_client:
        agent = drag.MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            allowed_tools=("text.word_count",),
            max_steps=3,
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "Use text.word_count to count the words in the phrase "
                "'design research workshop', then finish with only word_count."
            ),
            request_id=request_id,
        )
    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
