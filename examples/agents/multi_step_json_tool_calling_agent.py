"""# Agents / Multi Step JSON Tool Calling Agent.

## Introduction
Toolformer motivates tool-use planning, JSON Schema defines stable machine-readable contracts, and OpenAI
function-calling guidance captures operational patterns for structured tool dispatch. This example shows a
JSON-mode agent that repeatedly selects tools through explicit schema-constrained payloads.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["MultiStepAgent.run(...)"]
    C --> D["WorkflowRuntime loop enforces continuation and max-step policy"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<example-specific payload>",
     "terminated_reason": "<string-or-null>",
     "error": null,
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }

## References
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from design_research_agents import CallableToolConfig, LlamaCppServerLLMClient, MultiStepAgent, Toolbox, Tracer

_JSON_ALLOWED_TOOLS: tuple[str, ...] = ("repo.readme_snapshot",)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_STRONGER_LLAMA_CLIENT_KWARGS = {
    "model": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "hf_model_repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "api_model": "qwen3-4b-instruct-2507-q4km",
    "context_window": 8192,
    "startup_timeout_seconds": 180.0,
}


def _readme_snapshot(payload: Mapping[str, object]) -> dict[str, object]:
    """Return compact README metadata with no model-supplied path handling."""
    del payload
    readme_path = Path("README.md")
    readme_text = readme_path.read_text(encoding="utf-8")
    lines = readme_text.splitlines()
    first_heading = next((line.lstrip("#").strip() for line in lines if line.startswith("#")), "")
    return {
        "path": str(readme_path),
        "line_count": len(lines),
        "first_heading": first_heading,
        # The second step should copy these fields into the built-in final_answer action.
        "terminal_payload": {"result": True},
        "terminal_reason": "finish after one tool step",
    }


def main() -> None:
    """Execute one traced multi-step JSON tool-calling run."""
    # Stable ids make trace correlation and docs output easier to audit.
    request_id = "example-multi-step-json-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    # Pin the tool workspace root so README.md resolves consistently outside the repo root.
    with (
        Toolbox(
            workspace_root=_WORKSPACE_ROOT,
            enable_core_tools=False,
            callable_tools=(
                CallableToolConfig(
                    name="repo.readme_snapshot",
                    description="Return README line-count and first heading.",
                    handler=_readme_snapshot,
                ),
            ),
        ) as tool_runtime,
        LlamaCppServerLLMClient(**_STRONGER_LLAMA_CLIENT_KWARGS) as llm_client,
    ):
        agent = MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=2,
            # Constrain selection so the example exercises an explicit tool surface.
            allowed_tools=_JSON_ALLOWED_TOOLS,
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "When Current step is 1, repo.readme_snapshot is the only valid action. "
                "Do not use final_answer on step 1 because there is no prior observation yet. "
                "When Current step is 2, final_answer is the only valid action. "
                "Use tool_input exactly equal to the prior observation's terminal_payload object. "
                "Use reason exactly equal to the prior observation's terminal_reason string. "
                "Do not call repo.readme_snapshot again on step 2. "
                "Do not return an empty tool_input object."
            ),
            request_id=request_id,
        )

    summary = result.summary()
    expected_final_output = {"result": True}
    payload_matches_expected = summary.get("final_output") == expected_final_output
    error = summary.get("error")
    if not payload_matches_expected and error is None:
        error = "Expected final_output to copy terminal_payload exactly."
    rendered_summary = {
        **summary,
        "success": bool(summary.get("success")) and payload_matches_expected,
        "error": error,
        "expected_final_output": expected_final_output,
        "payload_matches_expected": payload_matches_expected,
    }
    print(json.dumps(rendered_summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
