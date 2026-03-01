"""# Clients / HTML LLM Client.

## Introduction
This example demonstrates the zero-dependency HTML stand-in client that ships with the framework. It is
useful for quickstarts, offline teaching, and trace smoke checks because it exercises the normal client
contract without depending on provider SDKs, local model runtimes, or network access.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``HTMLLLMClient.generate(...)`` with a fixed
   ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic inspection.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["HTMLLLMClient.generate(...)"]
    C --> D["LLMRequest/LLMResponse contracts stay unchanged"]
    C --> E["Tracer JSONL + console events"]
    D --> F["JSON payload"]
    E --> F
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "backend": {
       "base_url": null,
       "default_model": "html-standin-v1",
       "kind": "html",
       "max_retries": 0,
       "model_patterns": [
         "html-standin-v1"
       ],
       "name": "html-model"
     },
     "capabilities": {
       "json_mode": "none",
       "max_context_tokens": null,
       "streaming": true,
       "tool_calling": "none",
       "vision": false
     },
     "client_class": "HTMLLLMClient",
     "default_model": "html-standin-v1",
     "example": "clients/html_llm_client.py",
     "llm_call": {
       "prompt": "Wrap this design note in the stand-in HTML response.",
       "response_has_text": true,
       "response_model": "html-standin-v1",
       "response_provider": "html-model"
     },
     "server": null,
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }


## References
- `WHATWG HTML Living Standard <https://html.spec.whatwg.org/>`_
- `Python dataclasses <https://docs.python.org/3/library/dataclasses.html>`_
- `Python Protocols and structural subtyping <https://typing.python.org/en/latest/reference/protocols.html>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import HTMLLLMClient, Tracer
from design_research_agents.llm import LLMMessage, LLMRequest


def _build_payload() -> dict[str, object]:
    """Build the traced HTML client demo payload."""
    with HTMLLLMClient() as client:
        description = client.describe()
        prompt = "Wrap this design note in the stand-in HTML response."
        response = client.generate(
            LLMRequest(
                messages=(
                    LLMMessage(role="system", content="You emit deterministic HTML for demos."),
                    LLMMessage(role="user", content=prompt),
                ),
                model=client.default_model(),
            )
        )
        return {
            "client_class": description["client_class"],
            "default_model": description["default_model"],
            "llm_call": {
                "prompt": prompt,
                "response_text": response.text,
                "response_model": response.model,
                "response_provider": response.provider,
                "response_has_text": bool(response.text.strip()),
            },
            "backend": description["backend"],
            "capabilities": description["capabilities"],
            "server": description["server"],
        }


def main() -> None:
    """Run the traced HTML client demo payload."""
    request_id = "example-clients-html-llm-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesHTMLLLMClientCall",
        request_id=request_id,
        input_payload={"scenario": "html-llm-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/html_llm_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
