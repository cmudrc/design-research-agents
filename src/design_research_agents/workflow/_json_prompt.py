"""JSON prompt-workflow builders for model-backed study agents."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents._contracts import LLMMessage, LLMRequest, LLMResponse, ModelStep
from design_research_agents._contracts._llm import LLMClient

from .workflow import Workflow

_DEFAULT_SYSTEM_PROMPT = (
    "You are a careful study participant. Return valid JSON only and match the requested schema exactly."
)


def build_json_prompt_workflow(
    *,
    llm_client: LLMClient,
    response_schema: Mapping[str, object],
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    step_id: str = "json_response",
    temperature: float = 0.0,
    max_tokens: int | None = 400,
    request_metadata: Mapping[str, object] | None = None,
    default_request_id_prefix: str = "json-prompt-workflow",
    model: str | None = None,
    fallback_model_name: str | None = None,
    fallback_provider: str | None = None,
) -> Workflow:
    """Build a prompt-mode workflow that returns one parsed JSON output.

    Args:
        llm_client: LLM client used by the workflow's model step.
        response_schema: JSON schema supplied to the model request and workflow output.
        system_prompt: System message used to request strict JSON output.
        step_id: Stable model-step identifier.
        temperature: Sampling temperature passed to the LLM request.
        max_tokens: Maximum output tokens passed to the LLM request.
        request_metadata: Metadata added to every LLM request from this workflow.
        default_request_id_prefix: Prefix used when workflow callers omit a request id.
        model: Optional model override passed on the request.
        fallback_model_name: Model label used in output events when the response omits it.
        fallback_provider: Provider label used in output events when the response omits it.

    Returns:
        Workflow containing one model step that parses JSON into ``final_output``.
    """
    schema = dict(response_schema)
    metadata = dict(request_metadata or {})

    def request_builder(context: Mapping[str, object]) -> LLMRequest:
        """Build one structured JSON request from workflow context."""
        return LLMRequest(
            messages=(
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=str(context["prompt"])),
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=schema,
            metadata=metadata,
        )

    def response_parser(response: LLMResponse, _context: Mapping[str, object]) -> dict[str, object]:
        """Parse one LLM response into workflow output, metrics, and events."""
        model_text = _strip_markdown_fences(response.text.strip())
        parsed_json = _parse_json_value(model_text)
        provider = str(response.provider or fallback_provider or "")
        model_name = str(response.model or fallback_model_name or "")
        return {
            "final_output": parsed_json,
            "metrics": _usage_metrics(response.usage),
            "events": [
                {
                    "event_type": "model_response",
                    "actor_id": "agent",
                    "text": model_text,
                    "meta_json": {"provider": provider, "model_name": model_name},
                }
            ],
        }

    return Workflow(
        steps=(
            ModelStep(
                step_id=step_id,
                llm_client=llm_client,
                request_builder=request_builder,
                response_parser=response_parser,
            ),
        ),
        output_schema=schema,
        default_request_id_prefix=default_request_id_prefix,
    )


def _strip_markdown_fences(text: str) -> str:
    """Strip one optional fenced-code wrapper from model output."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_value(text: str) -> object:
    """Parse a JSON value, falling back to the first embedded object or array."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return parsed
    raise ValueError("Expected model response to contain valid JSON.")


def _usage_metrics(usage: object) -> dict[str, object]:
    """Normalize optional usage payloads into canonical metric names."""
    metrics: dict[str, object] = {"cost_usd": 0.0}
    if isinstance(usage, Mapping):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
    if isinstance(prompt_tokens, int):
        metrics["input_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        metrics["output_tokens"] = completion_tokens
    return metrics


__all__ = ["build_json_prompt_workflow"]
