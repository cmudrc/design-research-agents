"""# Agents / VS Code Hello World.

## Introduction
This example is intentionally self-contained so the VS Code launch configuration can run on a fresh checkout
without a live model server. It still exercises the public ``DirectLLMCall`` API, which makes it useful for
verifying that the virtual environment, debugger, editable install, and ``PYTHONPATH`` wiring are all correct.


## Technical Implementation
```mermaid
flowchart LR
    A["Press F5 in VS Code"] --> B["vscode_hello_world.py"]
    B --> C["_HelloWorldLLMClient.generate(...)"]
    B --> D["DirectLLMCall.run(...)"]
    C --> D
    D --> E["ExecutionResult.summary()"]
    E --> F["JSON output in integrated terminal"]
```

1. Define a tiny local client that implements ``generate(...)`` and returns a deterministic ``LLMResponse``.
2. Construct ``DirectLLMCall`` using only the public top-level package API.
3. Run one prompt through the direct-call path and collect the normalized execution summary.
4. Print JSON output so the VS Code debugger and terminal show one obvious success signal.


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "Hello from the VS Code onboarding example.",
     "terminated_reason": null,
     "error": null
   }


## References
- `VS Code Python environments <https://code.visualstudio.com/docs/python/environments>`_
- `Python virtual environments <https://docs.python.org/3/library/venv.html>`_
- `VS Code debugging <https://code.visualstudio.com/docs/debugtest/debugging>`_
"""

from __future__ import annotations

import json

import design_research_agents as drag


class _HelloWorldLLMClient:
    """Minimal deterministic client used by the VS Code onboarding example."""

    def generate(self, request: drag.LLMRequest) -> drag.LLMResponse:
        del request
        return drag.LLMResponse(
            text="Hello from the VS Code onboarding example.",
            model=self.default_model(),
            provider="local-vscode-stub",
        )

    def default_model(self) -> str:
        return "vscode-local-stub"

    def close(self) -> None:
        return None

    def __enter__(self) -> _HelloWorldLLMClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


def main() -> None:
    """Run one deterministic direct-LLM example for VS Code onboarding."""
    namespaces = (
        drag.integration.__name__,
        drag.model_selection.__name__,
        drag.study.__name__,
    )
    print(f"design-research-agents {drag.__version__}; namespaces={namespaces}")

    with _HelloWorldLLMClient() as llm_client:
        agent = drag.DirectLLMCall(
            llm_client=llm_client,
            system_prompt="You are a friendly onboarding assistant.",
        )
        result = agent.run(
            prompt="Say hello to a new design research teammate.",
            request_id="example-vscode-hello-world-001",
        )

    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
